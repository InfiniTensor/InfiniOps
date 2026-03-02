#ifndef INFINI_OPS_CUDA_RMS_NORM_KERNEL_H_
#define INFINI_OPS_CUDA_RMS_NORM_KERNEL_H_

#include <cstdint>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "base/rms_norm.h"
#include "cuda/rms_norm/kernel.cuh"
#include "data_type.h"
#include "dispatcher.h"

namespace infini::ops {

namespace {

constexpr unsigned int kBlockSize = 256;

}  // namespace

template <typename Backend>
class CudaRmsNorm : public RmsNorm {
 public:
  CudaRmsNorm(const Tensor y, const Tensor x, const Tensor w, float epsilon)
      : RmsNorm{y, x, w, epsilon} {}

  CudaRmsNorm(const Tensor y, const Tensor x, const Tensor w)
      : CudaRmsNorm{y, x, w, 1e-6f} {}

  void operator()(void* stream, Tensor y, const Tensor x, const Tensor w,
                  float /*epsilon*/) const override {
    auto cuda_stream =
        static_cast<typename Backend::stream_t>(stream ? stream : 0);

    if constexpr (Backend::needs_device_set) {
      cudaSetDevice(y.device().index());
    }

    auto stride_x_batch = x_strides_.size() > 1 ? x_strides_[0] : 0;
    auto stride_x_nhead = x_strides_.size() > 1 ? x_strides_[1] : x_strides_[0];
    auto stride_y_batch = y_strides_.size() > 1 ? y_strides_[0] : 0;
    auto stride_y_nhead = y_strides_.size() > 1 ? y_strides_[1] : y_strides_[0];

    uint32_t num_blocks = static_cast<uint32_t>(batch_size_ * nhead_);

    if (y.dtype() != x.dtype() || y.dtype() != w.dtype()) {
      std::abort();
    }

    DispatchFunc<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>(
        y.dtype(),
        [&]<typename T>() {
          rmsnormKernel<kBlockSize, float, T, T>
              <<<num_blocks, kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(y.data()), stride_y_batch,
                  stride_y_nhead, reinterpret_cast<const T*>(x.data()),
                  stride_x_batch, stride_x_nhead,
                  reinterpret_cast<const T*>(w.data()), nhead_, dim_, epsilon_);
        },
        "CudaRmsNorm::operator()");

    if constexpr (Backend::needs_stream_sync) {
      cudaStreamSynchronize(cuda_stream);
    }
  }
};

}  // namespace infini::ops

#endif
