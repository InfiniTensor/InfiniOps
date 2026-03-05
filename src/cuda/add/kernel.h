#ifndef INFINI_OPS_CUDA_ADD_KERNEL_H_
#define INFINI_OPS_CUDA_ADD_KERNEL_H_

#include <utility>

#include "base/add.h"
#include "common/cuda/kernel_commons.h"
#include "common/generic_utils.h"

namespace infini::ops {

typedef struct AddOp {
 public:
  static constexpr std::size_t num_inputs = 2;
  template <typename T>
  __device__ __forceinline__ T operator()(const T& input,
                                          const T& other) const {
    if constexpr (std::is_same_v<T, half2>) {
      return __hadd2(input, other);
    } else if constexpr (std::is_same_v<T, half> ||
                         std::is_same_v<T, TypeMapType<DataType::kBFloat16>>) {
      return __hadd(input, other);
    } else if constexpr (std::is_same_v<T, float>) {
      return __fadd_rn(input, other);
    } else {
      return input + other;
    }
  }
} AddOp;

template <typename T>
__global__ void AddKernel(
    T* out, const T* input, const T* other, const Tensor::Size* out_shape,
    const Tensor::Size* input_shape, const Tensor::Size* other_shape,
    const Tensor::Stride* out_strides, const Tensor::Stride* input_strides,
    const Tensor::Stride* other_strides, size_t output_size, size_t ndim,
    size_t offset, bool out_contiguous, bool input_contiguous,
    bool other_contiguous) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

  if (idx < output_size) {
    Tensor::Size out_idx =
        out_contiguous ? idx : indexToOffset(idx, ndim, out_shape, out_strides);
    Tensor::Size input_idx =
        input_contiguous ? idx
                         : indexToOffset(idx, ndim, input_shape, input_strides);
    Tensor::Size other_idx =
        other_contiguous ? idx
                         : indexToOffset(idx, ndim, other_shape, other_strides);

    out[out_idx] = AddOp{}(input[input_idx], other[other_idx]);
  }
}

template <typename Backend>
class CudaAdd : public Add {
 public:
  CudaAdd(const Tensor input, const Tensor other, Tensor out)
      : Add{input, other, out} {
    size_t shape_size = ndim_ * sizeof(*d_input_shape_);
    size_t strides_size = ndim_ * sizeof(*d_input_strides_);

    Backend::malloc((void**)&d_input_shape_, shape_size);
    Backend::malloc((void**)&d_other_shape_, shape_size);
    Backend::malloc((void**)&d_out_shape_, shape_size);
    Backend::malloc((void**)&d_input_strides_, strides_size);
    Backend::malloc((void**)&d_other_strides_, strides_size);
    Backend::malloc((void**)&d_out_strides_, strides_size);

    Backend::memcpy(d_input_shape_, input_shape_.data(), shape_size,
                    Backend::memcpyH2D);
    Backend::memcpy(d_other_shape_, other_shape_.data(), shape_size,
                    Backend::memcpyH2D);
    Backend::memcpy(d_out_shape_, out_shape_.data(), shape_size,
                    Backend::memcpyH2D);
    Backend::memcpy(d_input_strides_, input_strides_.data(), strides_size,
                    Backend::memcpyH2D);
    Backend::memcpy(d_other_strides_, other_strides_.data(), strides_size,
                    Backend::memcpyH2D);
    Backend::memcpy(d_out_strides_, out_strides_.data(), strides_size,
                    Backend::memcpyH2D);
  }

  ~CudaAdd() {
    Backend::free(d_input_shape_);
    Backend::free(d_other_shape_);
    Backend::free(d_out_shape_);
    Backend::free(d_input_strides_);
    Backend::free(d_other_strides_);
    Backend::free(d_out_strides_);
  }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    DispatchFunc<AllTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          // TODO(lzm): currently hard-code block_size to be 256.
          dim3 blockDims(
              std::min(static_cast<Tensor::Size>(256), output_size_));
          dim3 gridDims(utils::CeilDiv(output_size_, blockDims.x));
          size_t step = gridDims.x * blockDims.x;

          T* d_out = reinterpret_cast<T*>(out.data());
          const T* d_input = reinterpret_cast<const T*>(input.data());
          const T* d_other = reinterpret_cast<const T*>(other.data());

          for (size_t i = 0; i < output_size_; i += step) {
            AddKernel<<<gridDims, blockDims, 0,
                        static_cast<typename Backend::stream_t>(stream_)>>>(
                d_out, d_input, d_other, d_out_shape_, d_input_shape_,
                d_other_shape_, d_out_strides_, d_input_strides_,
                d_other_strides_, output_size_, ndim_, i, is_out_contiguous_,
                is_input_contiguous_, is_other_contiguous_);
          }
        },
        "CudaAdd::operator()");
  }

 private:
  Tensor::Size* d_input_shape_{nullptr};

  Tensor::Size* d_other_shape_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_other_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
