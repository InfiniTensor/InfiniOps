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
  using RmsNorm::RmsNorm;

  void operator()(void* stream, const Tensor input, const Tensor weight,
                  float eps, Tensor out) const override {
    auto cuda_stream =
        static_cast<typename Backend::stream_t>(stream ? stream : 0);

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

    DispatchFunc<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>(
        out.dtype(),
        [&]<typename T>() {
          RmsNormKernel<kBlockSize, float, T, T>
              <<<num_blocks, kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()), stride_out_batch,
                  stride_out_nhead, reinterpret_cast<const T*>(input.data()),
                  stride_input_batch, stride_input_nhead,
                  reinterpret_cast<const T*>(weight.data()), nhead_, dim_,
                  eps_);
        },
        "CudaRmsNorm::operator()");
  }
};

}  // namespace infini::ops

#endif
