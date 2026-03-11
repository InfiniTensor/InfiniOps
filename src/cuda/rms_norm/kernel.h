#ifndef INFINI_OPS_CUDA_RMS_NORM_KERNEL_H_
#define INFINI_OPS_CUDA_RMS_NORM_KERNEL_H_

#include <cstdint>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "base/rms_norm.h"
#include "common/cuda/kernel_commons.h"
#include "cuda/rms_norm/kernel.cuh"
#include "data_type.h"
#include "dispatcher.h"

namespace infini::ops {

template <typename Backend>
class CudaRmsNorm : public RmsNorm {
 public:
  using RmsNorm::RmsNorm;

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    auto cuda_stream =
        static_cast<typename Backend::stream_t>(stream_ ? stream_ : 0);

    auto stride_input_batch = input_strides_.size() > 1 ? input_strides_[0] : 0;
    auto stride_input_nhead =
        input_strides_.size() > 1 ? input_strides_[1] : input_strides_[0];
    auto stride_out_batch = out_strides_.size() > 1 ? out_strides_[0] : 0;
    auto stride_out_nhead =
        out_strides_.size() > 1 ? out_strides_[1] : out_strides_[0];

    uint32_t num_blocks = static_cast<uint32_t>(batch_size_ * nhead_);

    if (out.dtype() != input.dtype() || out.dtype() != weight.dtype()) {
      std::abort();
    }

    int block_size = GetOptimalBlockSize();

    DispatchFunc<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>(
        out.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;

#define LAUNCH_RMS_NORM_KERNEL(BLOCK_SIZE)                            \
  RmsNormKernel<BLOCK_SIZE, float, T, T>                              \
      <<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(                   \
          reinterpret_cast<T*>(out.data()), stride_out_batch,         \
          stride_out_nhead, reinterpret_cast<const T*>(input.data()), \
          stride_input_batch, stride_input_nhead,                     \
          reinterpret_cast<const T*>(weight.data()), nhead_, dim_, eps_);

          if (block_size == CUDA_BLOCK_SIZE_2048) {
            LAUNCH_RMS_NORM_KERNEL(CUDA_BLOCK_SIZE_2048)
          } else if (block_size == CUDA_BLOCK_SIZE_1024) {
            LAUNCH_RMS_NORM_KERNEL(CUDA_BLOCK_SIZE_1024)
          } else if (block_size == CUDA_BLOCK_SIZE_512) {
            LAUNCH_RMS_NORM_KERNEL(CUDA_BLOCK_SIZE_512)
          } else if (block_size == CUDA_BLOCK_SIZE_256) {
            LAUNCH_RMS_NORM_KERNEL(CUDA_BLOCK_SIZE_256)
          } else {
            LAUNCH_RMS_NORM_KERNEL(CUDA_BLOCK_SIZE_128)
          }

#undef LAUNCH_RMS_NORM_KERNEL
        },
        "CudaRmsNorm::operator()");
  }
};

}  // namespace infini::ops

#endif
