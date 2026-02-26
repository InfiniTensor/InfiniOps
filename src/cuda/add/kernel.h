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
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    if constexpr (std::is_same_v<T, half2>) {
      return __hadd2(a, b);
    } else if constexpr (std::is_same_v<T, half> ||
                         std::is_same_v<T, TypeMapType<DataType::kBFloat16>>) {
      return __hadd(a, b);
    } else if constexpr (std::is_same_v<T, float>) {
      return __fadd_rn(a, b);
    } else {
      return a + b;
    }
  }
} AddOp;

template <typename T>
__global__ void AddKernel(
    T* c, const T* a, const T* b, const Tensor::Size* c_shape,
    const Tensor::Size* a_shape, const Tensor::Size* b_shape,
    const Tensor::Stride* c_strides, const Tensor::Stride* a_strides,
    const Tensor::Stride* b_strides, size_t output_size, size_t ndim,
    size_t offset, bool c_contiguous, bool a_contiguous, bool b_contiguous) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

  if (idx < output_size) {
    Tensor::Size c_idx =
        c_contiguous ? idx : indexToOffset(idx, ndim, c_shape, c_strides);
    Tensor::Size a_idx =
        a_contiguous ? idx : indexToOffset(idx, ndim, a_shape, a_strides);
    Tensor::Size b_idx =
        b_contiguous ? idx : indexToOffset(idx, ndim, b_shape, b_strides);

    c[c_idx] = AddOp{}(a[a_idx], b[b_idx]);
  }
}

template <typename Backend>
class CudaAdd : public Add {
 public:
  CudaAdd(const Tensor a, const Tensor b, Tensor c) : Add{a, b, c} {
    size_t shape_size = ndim_ * sizeof(*d_a_shape_);
    size_t strides_size = ndim_ * sizeof(*d_a_strides_);

    Backend::malloc((void**)&d_a_shape_, shape_size);
    Backend::malloc((void**)&d_b_shape_, shape_size);
    Backend::malloc((void**)&d_c_shape_, shape_size);
    Backend::malloc((void**)&d_a_strides_, strides_size);
    Backend::malloc((void**)&d_b_strides_, strides_size);
    Backend::malloc((void**)&d_c_strides_, strides_size);

    Backend::memcpy(d_a_shape_, a_shape_.data(), shape_size,
                    Backend::MemcpyH2D);
    Backend::memcpy(d_b_shape_, b_shape_.data(), shape_size,
                    Backend::MemcpyH2D);
    Backend::memcpy(d_c_shape_, c_shape_.data(), shape_size,
                    Backend::MemcpyH2D);
    Backend::memcpy(d_a_strides_, a_strides_.data(), strides_size,
                    Backend::MemcpyH2D);
    Backend::memcpy(d_b_strides_, b_strides_.data(), strides_size,
                    Backend::MemcpyH2D);
    Backend::memcpy(d_c_strides_, c_strides_.data(), strides_size,
                    Backend::MemcpyH2D);
  }

  ~CudaAdd() {
    Backend::free(d_a_shape_);
    Backend::free(d_b_shape_);
    Backend::free(d_c_shape_);
    Backend::free(d_a_strides_);
    Backend::free(d_b_strides_);
    Backend::free(d_c_strides_);
  }

  void operator()(void* stream, const Tensor a, const Tensor b,
                  Tensor c) const override {
    DispatchFunc<FloatTypes>(
        c_type_,
        [&]<typename T>() {
          // TODO(lzm): currently hard-code block_size to be 256.
          dim3 blockDims(
              std::min(static_cast<Tensor::Size>(256), output_size_));
          dim3 gridDims(utils::CeilDiv(output_size_, blockDims.x));
          size_t step = gridDims.x * blockDims.x;

          T* d_c = reinterpret_cast<T*>(c.data());
          const T* d_a = reinterpret_cast<const T*>(a.data());
          const T* d_b = reinterpret_cast<const T*>(b.data());

          for (size_t i = 0; i < output_size_; i += step) {
            AddKernel<<<gridDims, blockDims, 0,
                        static_cast<typename Backend::stream_t>(stream)>>>(
                d_c, d_a, d_b, d_c_shape_, d_a_shape_, d_b_shape_, d_c_strides_,
                d_a_strides_, d_b_strides_, output_size_, ndim_, i,
                is_c_contiguous_, is_a_contiguous_, is_b_contiguous_);
          }
        },
        "CudaAdd::operator()");
  }

 private:
  Tensor::Size* d_a_shape_{nullptr};
  Tensor::Size* d_b_shape_{nullptr};
  Tensor::Size* d_c_shape_{nullptr};
  Tensor::Stride* d_a_strides_{nullptr};
  Tensor::Stride* d_b_strides_{nullptr};
  Tensor::Stride* d_c_strides_{nullptr};
};

}  // namespace infini::ops

#endif
