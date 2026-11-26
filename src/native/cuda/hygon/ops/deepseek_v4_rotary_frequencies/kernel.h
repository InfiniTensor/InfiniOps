#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>
#include "base/deepseek_v4_rotary_frequencies.h"
#pragma once
#include <cuda_runtime.h>
namespace infini::ops::v4_frequency_detail {
// Each tensor operation in table construction rounds separately in the reference.
static __device__ __forceinline__ float round_f32(float x) {volatile float value=x;return value;}
static __global__ void generate(float* out,size_t positions,size_t dim,float theta,
                                 bool scaled,float low,float high,float factor,bool split) {
 for(size_t k=blockIdx.x*blockDim.x+threadIdx.x;k<positions*(dim/2);k+=size_t(gridDim.x)*blockDim.x) {
  const auto pos=k/(dim/2),i=k%(dim/2);
  const float power=round_f32(powf(theta,float(2*i)/float(dim)));
  float frequency=round_f32(1.f/power);
  if(scaled){
   const float width=high==low?0.001f:high-low;
   const float reciprocal=round_f32(1.f/width);
   const float ramp=fminf(fmaxf(round_f32(round_f32(float(i)-low)*reciprocal),0.f),1.f);
   const float smooth=round_f32(1.f-ramp);
   const float first=round_f32(round_f32(frequency/factor)*round_f32(1.f-smooth));
   const float second=round_f32(frequency*smooth);
   frequency=round_f32(first+second);
  }
  const float angle=round_f32(float(pos)*frequency);
  out[pos*dim+(split?i:2*i)]=cosf(angle);
  out[pos*dim+(split?i+dim/2:2*i+1)]=sinf(angle);
 }
}
}

namespace infini::ops {
template<>
class Operator<DeepseekV4RotaryFrequencies,Device::Type::kHygon> : public DeepseekV4RotaryFrequencies {
 public:
  static bool supports(Tensor out,double theta,bool,double low,double high,double factor,bool) {
    return out.device().type()==Device::Type::kHygon && out.dtype()==DataType::kFloat32 &&
        out.ndim()==2 && out.size(0)>0 && out.size(1)>0 && out.size(1)%2==0 &&
        out.IsContiguous() && std::isfinite(theta) && theta>1 && theta<=std::numeric_limits<float>::max() &&
        std::isfinite(low) && std::isfinite(high) && high>=low &&
        std::isfinite(factor) && factor>=1 && factor<=std::numeric_limits<float>::max();
  }
  Operator(Tensor out,double theta,bool scaled,double low,double high,double factor,bool split)
      : DeepseekV4RotaryFrequencies(out,theta,scaled,low,high,factor,split) {
    if(!supports(out,theta,scaled,low,high,factor,split))throw std::invalid_argument("Invalid V4 rotary frequencies");
  }
  bool capture_safe() const override {return true;}
  void operator()(Tensor out,double theta,bool scaled,double low,double high,double factor,bool split) const override {
    if(!supports(out,theta,scaled,low,high,factor,split))throw std::invalid_argument("Invalid V4 rotary frequency execution");
    const size_t pairs=out.numel()/2;
    const auto grid=static_cast<unsigned>(std::min<size_t>((pairs-1)/256+1,65535));
    v4_frequency_detail::generate<<<grid,256,0,static_cast<cudaStream_t>(stream_)>>>(
        static_cast<float*>(out.data()),out.size(0),out.size(1),static_cast<float>(theta),
        scaled,static_cast<float>(low),static_cast<float>(high),static_cast<float>(factor),split);
    if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("V4 rotary frequency launch failed");
  }
};
} // namespace infini::ops
