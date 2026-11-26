#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "base/deepseek_v4_cast_rz.h"

namespace infini::ops {
namespace v4_cast_rz {
static __global__ void convert(const float* input, __nv_bfloat16* output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += static_cast<size_t>(gridDim.x) * blockDim.x)
    output[i] = __float2bfloat16_rz(input[i]);
}
}  // namespace v4_cast_rz

template<>
class Operator<DeepseekV4CastRz, Device::Type::kHygon> : public DeepseekV4CastRz {
 public:
  static bool supports(const Tensor input, const Tensor out) {
    return input.device().type() == Device::Type::kHygon &&
        out.device() == input.device() && input.dtype() == DataType::kFloat32 &&
        out.dtype() == DataType::kBFloat16 && input.shape() == out.shape() &&
        input.IsContiguous() && out.IsContiguous();
  }
  Operator(const Tensor input, Tensor out) : DeepseekV4CastRz(input,out) {
    if (!supports(input,out)) throw std::invalid_argument("Invalid DeepSeek V4 RZ cast");
  }
  bool capture_safe() const override { return true; }
  void operator()(const Tensor input, Tensor out) const override {
    if (!supports(input,out)) throw std::invalid_argument("Invalid DeepSeek V4 RZ cast execution");
    if (!input.numel()) return;
    const auto grid = static_cast<unsigned>(std::min<size_t>((input.numel()-1)/256+1,65535));
    v4_cast_rz::convert<<<grid,256,0,static_cast<cudaStream_t>(stream_)>>>(
        static_cast<const float*>(input.data()),static_cast<__nv_bfloat16*>(out.data()),input.numel());
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("DeepSeek V4 RZ cast launch failed");
  }
};
}  // namespace infini::ops
