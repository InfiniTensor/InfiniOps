#ifndef INFINI_OPS_CUDA_TEMPLATES_BINARY_ELEMENTWISE_CUH_
#define INFINI_OPS_CUDA_TEMPLATES_BINARY_ELEMENTWISE_CUH_

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

// Generic binary elementwise GPU kernel.
//
// `Op` is a device-side functor with signature `T operator()(const T&, const T&)`.
template <Device::Type kDev, typename Op, typename T, unsigned int BLOCK_SIZE>
__global__ void BinaryElementwiseKernel(
    T* __restrict__ out, const T* __restrict__ a, const T* __restrict__ b,
    const size_t* __restrict__ out_shape, const size_t* __restrict__ a_shape,
    const size_t* __restrict__ b_shape,
    const ptrdiff_t* __restrict__ out_strides,
    const ptrdiff_t* __restrict__ a_strides,
    const ptrdiff_t* __restrict__ b_strides, size_t output_size, size_t ndim,
    bool out_contig, bool a_contig, bool b_contig) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < output_size) {
    size_t out_idx =
        out_contig ? idx : IndexToOffset(idx, ndim, out_shape, out_strides);
    size_t a_idx =
        a_contig ? idx : IndexToOffset(idx, ndim, a_shape, a_strides);
    size_t b_idx =
        b_contig ? idx : IndexToOffset(idx, ndim, b_shape, b_strides);

    out[out_idx] = Op{}(a[a_idx], b[b_idx]);
  }
}

// Manages device metadata (shapes/strides) for a binary elementwise operator
// and provides a templated `Run` method for dtype-dispatched kernel launch.
template <typename Backend>
class BinaryElementwiseBrick {
 public:
  BinaryElementwiseBrick(const Tensor a, const Tensor b, const Tensor out,
                         Tensor::Size ndim) {
    size_t shape_bytes = ndim * sizeof(Tensor::Size);
    size_t stride_bytes = ndim * sizeof(Tensor::Stride);
    size_t total = 3 * (shape_bytes + stride_bytes);
    std::vector<std::byte> staging(total);

    Backend::Malloc((void**)&d_metadata_, total);

    size_t offset = 0;

    d_a_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(staging.data() + offset, a.shape().data(), shape_bytes);
    offset += shape_bytes;

    d_b_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(staging.data() + offset, b.shape().data(), shape_bytes);
    offset += shape_bytes;

    d_out_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(staging.data() + offset, out.shape().data(), shape_bytes);
    offset += shape_bytes;

    d_a_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(staging.data() + offset, a.strides().data(), stride_bytes);
    offset += stride_bytes;

    d_b_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(staging.data() + offset, b.strides().data(), stride_bytes);
    offset += stride_bytes;

    d_out_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(staging.data() + offset, out.strides().data(), stride_bytes);

    Backend::Memcpy(d_metadata_, staging.data(), total,
                    Backend::MemcpyHostToDevice);
  }

  ~BinaryElementwiseBrick() { Backend::Free(d_metadata_); }

  BinaryElementwiseBrick(const BinaryElementwiseBrick&) = delete;
  BinaryElementwiseBrick& operator=(const BinaryElementwiseBrick&) = delete;

  // Launch the elementwise kernel with dtype dispatch.
  //
  // `TypeList` is the compile-time list of supported `DataType` values
  // (e.g. `AllTypes`, `AllFloatTypes`).
  // `Op` is a device-side functor templated on `Device::Type kDev` with
  // a member `template <typename T> T operator()(const T&, const T&)`.
  template <typename TypeList, template <Device::Type> class Op>
  void Run(void* stream, const Tensor a, const Tensor b, Tensor out,
           Tensor::Size output_size, Tensor::Size ndim, bool a_contig,
           bool b_contig, bool out_contig, DataType dtype) const {
    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    DispatchFunc<TypeList, AllCudaBlockSizes>(
        {static_cast<int64_t>(dtype), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          auto cuda_stream =
              static_cast<typename Backend::Stream>(stream ? stream : 0);
          dim3 blockDims(
              std::min(static_cast<Tensor::Size>(block_size), output_size));
          dim3 gridDims(utils::CeilDiv(output_size, blockDims.x));

          BinaryElementwiseKernel<Backend::kDeviceType, Op<Backend::kDeviceType>,
                                  T, kBlockSize>
              <<<gridDims, blockDims, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()),
                  reinterpret_cast<const T*>(a.data()),
                  reinterpret_cast<const T*>(b.data()), d_out_shape_,
                  d_a_shape_, d_b_shape_, d_out_strides_, d_a_strides_,
                  d_b_strides_, output_size, ndim, out_contig, a_contig,
                  b_contig);
        },
        "BinaryElementwiseBrick::Run");
  }

 private:
  std::byte* d_metadata_{nullptr};

  Tensor::Size* d_a_shape_{nullptr};

  Tensor::Size* d_b_shape_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_a_strides_{nullptr};

  Tensor::Stride* d_b_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
