#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "base/deepseek_v4_mhc_post.h"
#include "residual_mix.cuh"

namespace infini::ops {
namespace v4_mhc_post {
static __global__ void mix(const __nv_bfloat16* activation,const __nv_bfloat16* residual,
                           const float* post,const float* matrix,__nv_bfloat16* out,
                           size_t n,size_t hidden) {
  for(size_t idx=blockIdx.x*blockDim.x+threadIdx.x;idx<n;idx+=gridDim.x*blockDim.x) {
    const size_t col=idx%hidden, stream=(idx/hidden)%4, row=idx/(hidden*4);
    // Preserve the residual-first FMA order used by the HCU reference.
    float value=matrix[row*16+stream]*__bfloat162float(residual[row*4*hidden+col]);
    value=fmaf(post[row*4+stream],__bfloat162float(activation[row*hidden+col]),value);
    for(unsigned i=1;i<4;++i)
      value=fmaf(matrix[row*16+i*4+stream],__bfloat162float(residual[(row*4+i)*hidden+col]),value);
    // Match the HCU mHC residual writeback used by SGLang.
    out[idx]=__float2bfloat16_rz(value);
  }
}
}
template<>
class Operator<DeepseekV4MhcPost,Device::Type::kHygon> : public DeepseekV4MhcPost {
 public:
  using DeepseekV4MhcPost::DeepseekV4MhcPost;
  bool capture_safe() const override {return true;}
  static bool supports(Tensor x,Tensor r,Tensor p,Tensor m,Tensor y) {
    if(r.ndim()!=3 || r.size(1)!=4 || !r.size(0) || !r.size(2) ||
       r.device().type()!=Device::Type::kHygon || r.dtype()!=DataType::kBFloat16 ||
       x.shape()!=Tensor::Shape{r.size(0),r.size(2)} || p.shape()!=Tensor::Shape{r.size(0),4} ||
       m.shape()!=Tensor::Shape{r.size(0),4,4} || y.shape()!=r.shape() ||
       y.dtype()!=r.dtype() || x.dtype()!=r.dtype() || p.dtype()!=DataType::kFloat32 ||
       m.dtype()!=DataType::kFloat32) return false;
    for(const auto& t:{x,r,p,m,y}) if(t.device()!=r.device() || !t.IsContiguous()) return false;
    return true;
  }
  void operator()(Tensor x,Tensor r,Tensor p,Tensor m,Tensor y) const override {
    if(!supports(x,r,p,m,y)) throw std::invalid_argument("Invalid DeepSeek V4 mHC post");
    // Reuse inputs across streams for prefill. M<4 retains the original kernel:
    // the extra occupancy is faster for the small decode shape on BW/gfx936.
    if (r.size(2) == 4096 && r.size(0) >= 4) {
      const auto elements = r.size(0) * r.size(2);
      v4_mhc_post_opt::mix_reuse<<<static_cast<unsigned>(std::min<size_t>((elements-1)/256+1,65535)),256,0,
                                 static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const __nv_bfloat16*>(x.data()),static_cast<const __nv_bfloat16*>(r.data()),
          static_cast<const float*>(p.data()),static_cast<const float*>(m.data()),
          static_cast<__nv_bfloat16*>(y.data()),r.size(0),r.size(2));
    } else {
      v4_mhc_post::mix<<<static_cast<unsigned>(std::min<size_t>((y.numel()-1)/128+1,65535)),128,0,
                         static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const __nv_bfloat16*>(x.data()),static_cast<const __nv_bfloat16*>(r.data()),
          static_cast<const float*>(p.data()),static_cast<const float*>(m.data()),
          static_cast<__nv_bfloat16*>(y.data()),y.numel(),r.size(2));
    }
    if(cudaGetLastError()!=cudaSuccess) throw std::runtime_error("DeepSeek V4 mHC post launch failed");
  }
};
} // namespace infini::ops
