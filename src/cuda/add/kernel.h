#ifndef INFINI_OPS_CUDA_ADD_KERNEL_H_
#define INFINI_OPS_CUDA_ADD_KERNEL_H_

#include <cstdint>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "base/add.h"
#include "common/generic_utils.h"
#include "cuda/add/kernel.cuh"

namespace infini::ops {

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
          auto cuda_stream =
              static_cast<typename Backend::stream_t>(stream_ ? stream_ : 0);
          int block_size = GetOptimalBlockSize();
          dim3 blockDims(
              std::min(static_cast<Tensor::Size>(block_size), output_size_));
          dim3 gridDims(utils::CeilDiv(output_size_, blockDims.x));
          size_t step = gridDims.x * blockDims.x;

          T* d_out = reinterpret_cast<T*>(out.data());
          const T* d_input = reinterpret_cast<const T*>(input.data());
          const T* d_other = reinterpret_cast<const T*>(other.data());

#define LAUNCH_ADD_KERNEL(BLOCK_SIZE)                                          \
  for (size_t i = 0; i < output_size_; i += step) {                            \
    AddKernel<T, BLOCK_SIZE><<<gridDims, blockDims, 0, cuda_stream>>>(         \
        d_out, d_input, d_other, d_out_shape_, d_input_shape_, d_other_shape_, \
        d_out_strides_, d_input_strides_, d_other_strides_, output_size_,      \
        ndim_, i, is_out_contiguous_, is_input_contiguous_,                    \
        is_other_contiguous_);                                                 \
  }

          if (block_size == CUDA_BLOCK_SIZE_2048) {
            LAUNCH_ADD_KERNEL(CUDA_BLOCK_SIZE_2048)
          } else if (block_size == CUDA_BLOCK_SIZE_1024) {
            LAUNCH_ADD_KERNEL(CUDA_BLOCK_SIZE_1024)
          } else if (block_size == CUDA_BLOCK_SIZE_512) {
            LAUNCH_ADD_KERNEL(CUDA_BLOCK_SIZE_512)
          } else if (block_size == CUDA_BLOCK_SIZE_256) {
            LAUNCH_ADD_KERNEL(CUDA_BLOCK_SIZE_256)
          } else {
            LAUNCH_ADD_KERNEL(CUDA_BLOCK_SIZE_128)
          }

#undef LAUNCH_ADD_KERNEL
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
