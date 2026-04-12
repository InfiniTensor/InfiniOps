#ifndef INFINI_OPS_CUDA_TEMPLATES_UNARY_ELEMENTWISE_CUH_
#define INFINI_OPS_CUDA_TEMPLATES_UNARY_ELEMENTWISE_CUH_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/generic_utils.h"
#include "cuda/kernel_commons.cuh"
#include "cuda/runtime_utils.h"
#include "dispatcher.h"
#include "tensor.h"

namespace infini::ops {

// Vectorized unary elementwise kernel for contiguous tensors.
//
// Uses vectorized load/store with grid-stride loop.  VEC_SIZE is chosen
// based on the *input* type to target 128-bit loads.
template <Device::Type kDev, typename Op, typename TIn, typename TOut,
          unsigned int BLOCK_SIZE, int VEC_SIZE>
__global__ void UnaryElementwiseVecKernel(TOut* __restrict__ out,
                                          const TIn* __restrict__ in,
                                          size_t output_size) {
  Op op{};
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  size_t vec_count = output_size / VEC_SIZE;

  using InVec = typename utils::AlignedVec<TIn, VEC_SIZE>::type;
  const InVec* in_vec = reinterpret_cast<const InVec*>(in);

  // Use output vectorization when sizeof matches (same type cast) or
  // when VEC_SIZE output elements fit naturally.
  using OutVec = typename utils::AlignedVec<TOut, VEC_SIZE>::type;
  OutVec* out_vec = reinterpret_cast<OutVec*>(out);

  for (size_t i = tid; i < vec_count; i += stride) {
    InVec vin = in_vec[i];
    const TIn* pin = reinterpret_cast<const TIn*>(&vin);
    OutVec vout;
    TOut* po = reinterpret_cast<TOut*>(&vout);

    #pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      po[j] = op.template operator()<TIn, TOut>(pin[j]);
    }

    out_vec[i] = vout;
  }

  // Handle remaining elements.
  size_t tail_start = vec_count * VEC_SIZE;

  for (size_t i = tail_start + tid; i < output_size; i += stride) {
    out[i] = op.template operator()<TIn, TOut>(in[i]);
  }
}

// Generic unary elementwise GPU kernel (non-contiguous path).
//
// `Op` is a device-side functor with signature `TOut operator()(const TIn&)`.
template <Device::Type kDev, typename Op, typename TIn, typename TOut,
          unsigned int BLOCK_SIZE>
__global__ void UnaryElementwiseKernel(
    TOut* __restrict__ out, const TIn* __restrict__ in,
    const size_t* __restrict__ out_shape, const size_t* __restrict__ in_shape,
    const ptrdiff_t* __restrict__ out_strides,
    const ptrdiff_t* __restrict__ in_strides, size_t output_size, size_t ndim,
    bool out_contig, bool in_contig) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < output_size) {
    size_t out_idx =
        out_contig ? idx : IndexToOffset(idx, ndim, out_shape, out_strides);
    size_t in_idx =
        in_contig ? idx : IndexToOffset(idx, ndim, in_shape, in_strides);

    out[out_idx] = Op{}.template operator()<TIn, TOut>(in[in_idx]);
  }
}

// Manages device metadata (shapes/strides) for a unary elementwise operator
// and provides a templated `Run` method for dual-dtype-dispatched kernel launch.
template <typename Backend>
class UnaryElementwiseBrick {
 public:
  UnaryElementwiseBrick(const Tensor input, Tensor out, Tensor::Size ndim) {
    size_t shape_bytes = ndim * sizeof(Tensor::Size);
    size_t stride_bytes = ndim * sizeof(Tensor::Stride);
    size_t total = 2 * (shape_bytes + stride_bytes);
    std::vector<std::byte> staging(total);

    Backend::Malloc((void**)&d_metadata_, total);

    size_t offset = 0;

    d_in_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(staging.data() + offset, input.shape().data(), shape_bytes);
    offset += shape_bytes;

    d_out_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(staging.data() + offset, out.shape().data(), shape_bytes);
    offset += shape_bytes;

    d_in_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(staging.data() + offset, input.strides().data(), stride_bytes);
    offset += stride_bytes;

    d_out_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(staging.data() + offset, out.strides().data(), stride_bytes);

    Backend::Memcpy(d_metadata_, staging.data(), total,
                    Backend::MemcpyHostToDevice);
  }

  ~UnaryElementwiseBrick() { Backend::Free(d_metadata_); }

  UnaryElementwiseBrick(const UnaryElementwiseBrick&) = delete;

  UnaryElementwiseBrick& operator=(const UnaryElementwiseBrick&) = delete;

  // Launch the elementwise kernel with dual-dtype dispatch.
  //
  // `InputTypeList` and `OutputTypeList` are the compile-time lists of
  // supported `DataType` values for input and output respectively.
  // `Op` is a device-side functor templated on `Device::Type kDev` with
  // a member `template <typename TIn, typename TOut> TOut operator()(const TIn&)`.
  template <typename InputTypeList, typename OutputTypeList,
            template <Device::Type> class Op>
  void Run(void* stream, const Tensor input, Tensor out,
           Tensor::Size output_size, Tensor::Size ndim, bool in_contig,
           bool out_contig, DataType input_dtype,
           DataType output_dtype) const {
    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    DispatchFunc<InputTypeList, OutputTypeList, AllCudaBlockSizes>(
        {static_cast<int64_t>(input_dtype), static_cast<int64_t>(output_dtype),
         block_size},
        [&](auto list_tag) {
          using TIn = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using TOut = TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
          constexpr int kBlockSize = ListGet<2>(list_tag);

          auto cuda_stream =
              static_cast<typename Backend::Stream>(stream ? stream : 0);

          if (in_contig && out_contig) {
            // Vectorized path: 128-bit loads on input type.
            constexpr int kVecSize = utils::OptimalVecSize<TIn>();
            size_t vec_count = output_size / kVecSize;
            size_t total_threads = vec_count > 0 ? vec_count : output_size;
            dim3 blockDims(std::min(static_cast<size_t>(block_size),
                                    total_threads));
            dim3 gridDims(
                std::min(utils::CeilDiv(total_threads, blockDims.x),
                         static_cast<decltype(total_threads)>(65535)));

            UnaryElementwiseVecKernel<Backend::kDeviceType,
                                      Op<Backend::kDeviceType>, TIn, TOut,
                                      kBlockSize, kVecSize>
                <<<gridDims, blockDims, 0, cuda_stream>>>(
                    reinterpret_cast<TOut*>(out.data()),
                    reinterpret_cast<const TIn*>(input.data()), output_size);
          } else {
            dim3 blockDims(
                std::min(static_cast<Tensor::Size>(block_size), output_size));
            dim3 gridDims(utils::CeilDiv(output_size, blockDims.x));

            UnaryElementwiseKernel<Backend::kDeviceType,
                                   Op<Backend::kDeviceType>, TIn, TOut,
                                   kBlockSize>
                <<<gridDims, blockDims, 0, cuda_stream>>>(
                    reinterpret_cast<TOut*>(out.data()),
                    reinterpret_cast<const TIn*>(input.data()), d_out_shape_,
                    d_in_shape_, d_out_strides_, d_in_strides_, output_size,
                    ndim, out_contig, in_contig);
          }
        },
        "UnaryElementwiseBrick::Run");
  }

 private:
  std::byte* d_metadata_{nullptr};

  Tensor::Size* d_in_shape_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_in_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
