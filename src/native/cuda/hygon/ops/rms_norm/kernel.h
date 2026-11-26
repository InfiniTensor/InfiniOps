#ifndef INFINI_OPS_HYGON_RMS_NORM_KERNEL_H_
#define INFINI_OPS_HYGON_RMS_NORM_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/rms_norm/kernel.h"

namespace infini::ops {
namespace hygon_rms_detail {

// Warp-per-head reduction for the FP32 512-wide attention head path.
static __global__ void NormHeads(const float* input, const float* weight,
                                float* output, size_t heads, size_t rows,
                                int64_t input_row_stride, int64_t input_head_stride,
                                int64_t output_row_stride, int64_t output_head_stride,
                                float epsilon) {
  const unsigned lane = threadIdx.x;
  const size_t head = blockIdx.x;
  const size_t row = head / heads;
  if (row >= rows) return;
  input += row * input_row_stride + (head % heads) * input_head_stride;
  output += row * output_row_stride + (head % heads) * output_head_stride;
  float squared_sum = 0.0f;
#pragma unroll
  for (unsigned tile = 0; tile < 2; ++tile) {
#pragma unroll
    for (unsigned element = 0; element < 8; ++element) {
      const float x = input[tile * 256 + lane * 8 + element];
      squared_sum = __fmaf_rn(x, x, squared_sum);
    }
  }
#pragma unroll
  for (unsigned offset = 16; offset; offset >>= 1)
    squared_sum += __shfl_xor_sync(0xffffffff, squared_sum, offset, 32);
  const float inverse_rms = rsqrtf(squared_sum / 512.0f + epsilon);
  for (unsigned i = lane; i < 512; i += 32)
    output[i] = __fmul_rn(__fmul_rn(input[i], inverse_rms), weight[i]);
}
static __global__ void NormKv(const float* input, const float* weight,
                             float* output, int64_t input_stride,
                             int64_t output_stride, float epsilon) {
  __shared__ float partial[8];
  const unsigned thread = threadIdx.x, lane = thread % 32;
  input += blockIdx.x * input_stride;
  output += blockIdx.x * output_stride;
  float sum = 0.0f;
#pragma unroll
  for (unsigned j=0;j<2;++j) {
    const float x=input[2*thread+j];
    sum=__fmaf_rn(x,x,sum);
  }
#pragma unroll
  for (unsigned offset=16;offset;offset>>=1)
    sum+=__shfl_xor_sync(0xffffffff,sum,offset,32);
  if (!lane) partial[thread/32]=sum;
  __syncthreads();
  sum=partial[lane%8];
#pragma unroll
  for (unsigned offset=4;offset;offset>>=1)
    sum+=__shfl_xor_sync(0xffffffff,sum,offset,8);
  const float inverse_rms=rsqrtf(sum/512.0f+epsilon);
#pragma unroll
  for (unsigned j=0;j<2;++j) {
    const unsigned i=2*thread+j;
    output[i]=__fmul_rn(__fmul_rn(input[i],inverse_rms),weight[i]);
  }
}
static __global__ __launch_bounds__(512) void NormHidden(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                                 __nv_bfloat16* output, int64_t input_stride,
                                 int64_t output_stride, size_t width, float epsilon) {
  __shared__ float partial[8];
  const unsigned thread = threadIdx.x;
  input += blockIdx.x * input_stride;
  output += blockIdx.x * output_stride;
  float squared_sum = 0.0f;
#pragma unroll
  for (unsigned j = 0; j < 8; ++j) {
    const float value = __bfloat162float(input[thread * 8 + j]);
    squared_sum = __fmaf_rn(value, value, squared_sum);
  }
#pragma unroll
  for (unsigned offset = 32; offset; offset >>= 1)
    squared_sum += __shfl_xor_sync(0xffffffff, squared_sum, offset, 64);
  if (thread % 64 == 0) partial[thread / 64] = squared_sum;
  __syncthreads();
  if (thread < 64) {
    float total = thread < blockDim.x/64 ? partial[thread] : 0.0f;
#pragma unroll
    for (unsigned offset = 32; offset; offset >>= 1)
      total += __shfl_xor_sync(0xffffffff, total, offset, 64);
    if (thread == 0) partial[0] = rsqrtf(total / static_cast<float>(width) + epsilon);
  }
  __syncthreads();
  const float inverse_rms = partial[0];
#pragma unroll
  for (unsigned j = 0; j < 8; ++j) {
    const unsigned index = thread * 8 + j;
    const float normalized = __fmul_rn(__bfloat162float(input[index]), inverse_rms);
    output[index] = __float2bfloat16_rn(__fmul_rn(normalized, __bfloat162float(weight[index])));
  }
}
}  // namespace hygon_rms_detail


template <>
class Operator<RmsNorm, Device::Type::kHygon>
    : public CudaRmsNorm<Runtime<Device::Type::kHygon>> {
 public:
  using CudaRmsNorm<Runtime<Device::Type::kHygon>>::CudaRmsNorm;

  void operator()(const Tensor input, const Tensor weight, float epsilon,
                  Tensor out) const override {
    if (input.ndim() == 3 && input.size(-1) == 512 &&
        input.dtype() == DataType::kFloat32 && weight.dtype() == DataType::kFloat32 &&
        NativeSupports(input, weight, out)) {
      const size_t heads = input.size(1), rows = input.size(0);
      if (!heads || !rows) return;
      if (heads == 1) {
        hygon_rms_detail::NormKv<<<rows,256,0,static_cast<cudaStream_t>(stream_)>>>(
            static_cast<const float*>(input.data()),static_cast<const float*>(weight.data()),
            static_cast<float*>(out.data()),input.stride(0),out.stride(0),epsilon);
        if (cudaGetLastError()!=cudaSuccess) throw std::runtime_error("Hygon KV RMSNorm launch failed");
        return;
      }
      hygon_rms_detail::NormHeads<<<rows * heads, 32, 0, static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const float*>(input.data()), static_cast<const float*>(weight.data()),
          static_cast<float*>(out.data()), heads, rows, input.stride(0), input.stride(1),
          out.stride(0), out.stride(1), epsilon);
      if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("Hygon attention head RMSNorm launch failed");
      return;
    }
    if (input.ndim() == 2 && (input.size(1) == 4096 || input.size(1) == 1024) &&
        input.dtype() == DataType::kBFloat16 && weight.dtype() == DataType::kBFloat16 &&
        NativeSupports(input, weight, out)) {
      if (!input.size(0)) return;
      hygon_rms_detail::NormHidden<<<input.size(0), input.size(1)/8, 0, static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const __nv_bfloat16*>(input.data()), static_cast<const __nv_bfloat16*>(weight.data()),
          static_cast<__nv_bfloat16*>(out.data()), input.stride(0), out.stride(0), input.size(1), epsilon);
      if (cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("Hygon hidden RMSNorm launch failed");
      return;
    }
    CudaRmsNorm<Runtime<Device::Type::kHygon>>::operator()(input, weight, epsilon, out);
  }

  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
