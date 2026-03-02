#ifndef INFINI_OPS_NVIDIA_RMS_NORM_CU_
#define INFINI_OPS_NVIDIA_RMS_NORM_CU_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda/rms_norm/kernel.cuh"
#include "data_type.h"
#include "nvidia/rms_norm/rms_norm.h"

namespace infini::ops {

namespace {

constexpr unsigned int kBlockSize = 256;

#define LAUNCH_RMSNORM_KERNEL(Tdata, Tweight, Tcompute)                        \
  rmsnormKernel<kBlockSize, Tcompute, Tdata, Tweight>                          \
      <<<num_blocks, kBlockSize, 0, cuda_stream>>>(                            \
          reinterpret_cast<Tdata*>(y.data()), stride_y_batch, stride_y_nhead,  \
          reinterpret_cast<const Tdata*>(x.data()), stride_x_batch,            \
          stride_x_nhead, reinterpret_cast<const Tweight*>(w.data()), nhead_,  \
          dim_, epsilon_)

}  // namespace

void Operator<RmsNorm, Device::Type::kNvidia>::operator()(
    void* stream, Tensor y, const Tensor x, const Tensor w,
    float /*epsilon*/) const {
  auto cuda_stream = static_cast<cudaStream_t>(stream);
  auto stride_x_batch = x_strides_.size() > 1 ? x_strides_[0] : 0;
  auto stride_x_nhead = x_strides_.size() > 1 ? x_strides_[1] : x_strides_[0];
  auto stride_y_batch = y_strides_.size() > 1 ? y_strides_[0] : 0;
  auto stride_y_nhead = y_strides_.size() > 1 ? y_strides_[1] : y_strides_[0];

  uint32_t num_blocks = static_cast<uint32_t>(batch_size_ * nhead_);

  if (y.dtype() != x.dtype() || y.dtype() != w.dtype()) {
    abort();
  }

  if (y.dtype() == DataType::kFloat32) {
    LAUNCH_RMSNORM_KERNEL(float, float, float);
  } else if (y.dtype() == DataType::kFloat16) {
    LAUNCH_RMSNORM_KERNEL(half, half, float);
  } else if (y.dtype() == DataType::kBFloat16) {
    LAUNCH_RMSNORM_KERNEL(__nv_bfloat16, __nv_bfloat16, float);
  } else {
    abort();
  }

#undef LAUNCH_RMSNORM_KERNEL
}

}  // namespace infini::ops

#endif
