#pragma once
#include <cuda_runtime.h>
#include "base/deepseek_v4_route.h"

namespace infini::ops {
namespace v4_route {

// Reproduce the Triton gate's stable log1p expansion and HCU math.
static __device__ float probability(float x) {
  const float exponent = __builtin_amdgcn_exp2f(-fabsf(x) * 1.4426950408889634f);
  volatile float shifted = 1.0f + exponent;
  const float log2 = __builtin_amdgcn_logf(shifted);
  const float ln2_hi = __uint_as_float(0x3f317217);
  const float ln2_lo = __uint_as_float(0x3377d1cf);
  const float product = __fmul_rn(log2, ln2_hi);
  float correction = fmaf(log2, ln2_hi, -product);
  correction = fmaf(log2, ln2_lo, correction);
  volatile float natural_log = fmaf(log2, ln2_hi, correction);
  volatile float numerator = exponent * natural_log;
  const float log1p = shifted == 1.0f ? exponent : numerator / (shifted - 1.0f);
  return __builtin_amdgcn_sqrtf(fmaxf(x, 0.0f) + log1p);
}

// HCU's hash gate uses stable sqrt(softplus), then a six-lane warp sum.
static __device__ float hash_probability(float x) {
  return sqrtf(fmaxf(x,0.f)+log1pf(expf(-fabsf(x))));
}

template<class Token>
__global__ void route(const float* logits,const float* bias,const Token* tokens,
                      const int64_t* hash,float scale,float* weights,int64_t* ids,
                      size_t rows,size_t experts,size_t k,size_t vocab) {
  for(size_t row=blockIdx.x*blockDim.x+threadIdx.x;row<rows;row+=gridDim.x*blockDim.x) {
    const auto* x=logits+row*experts;
    auto* selected=ids+row*k;
    auto* w=weights+row*k;
    const auto token=tokens ? static_cast<int64_t>(tokens[row]) : -1;
    float norm=0;
    for(size_t i=0;i<k;++i) {
      int64_t id=-1;
      if(hash) {
        if(token>=0 && size_t(token)<vocab) id=hash[size_t(token)*k+i];
      } else {
        float best=-INFINITY;
        for(size_t j=0;j<experts;++j) {
          const float rank=probability(x[j])+(bias ? bias[j] : 0);
          if(rank<=best) continue;
          bool used=false;
          for(size_t prev=0;prev<i;++prev) used|=selected[prev]==int64_t(j);
          if(!used) {best=rank; id=j;}
        }
      }
      const bool valid=id>=0 && size_t(id)<experts;
      selected[i]=valid ? id : -1;
      w[i]=valid ? (hash ? hash_probability(x[id]) : probability(x[id])) : 0.f;
      norm+=w[i];
    }
    if (k == 6) {
      // The two reference kernels pack the six weights into lanes differently.
      if (hash) {
        norm = ((w[0] + w[4]) + w[2]) + ((w[1] + w[5]) + w[3]);
      } else {
        volatile float first = (((w[0] + w[1]) + w[2]) + w[3]);
        volatile float second = w[4] + w[5];
        norm = first + second;
      }
    }
    for(size_t i=0;i<k;++i) w[i]=norm>0 ? (w[i]/norm)*scale : 0.f;
  }
}
} // namespace v4_route

template<>
class Operator<DeepseekV4Route,Device::Type::kHygon> : public DeepseekV4Route {
 public:
  using Opt=std::optional<Tensor>;
  using DeepseekV4Route::DeepseekV4Route;
  bool capture_safe() const override {return true;}
  static bool supports(Tensor x,Opt b,Opt t,Opt h,double s,Tensor w,Tensor i) {
    if(x.ndim()!=2 || !x.size(0) || !x.size(1) || w.ndim()!=2 || w.size(0)!=x.size(0) ||
       !w.size(1) || w.size(1)>x.size(1) || w.shape()!=i.shape() ||
       x.dtype()!=DataType::kFloat32 || w.dtype()!=DataType::kFloat32 || i.dtype()!=DataType::kInt64 ||
       x.device().type()!=Device::Type::kHygon || !std::isfinite(s) || s<=0 ||
       bool(t)!=bool(h) || (t && b)) return false;
    if(b && (b->shape()!=Tensor::Shape{x.size(1)} || b->dtype()!=DataType::kFloat32)) return false;
    if(t && (t->shape()!=Tensor::Shape{x.size(0)} ||
             (t->dtype()!=DataType::kInt32 && t->dtype()!=DataType::kInt64) ||
             h->ndim()!=2 || !h->size(0) || h->size(1)!=w.size(1) ||
             h->dtype()!=DataType::kInt64)) return false;
    for(const auto& a:{x,w,i}) if(a.device()!=x.device() || !a.IsContiguous()) return false;
    for(const auto& a:{b,t,h}) if(a && (a->device()!=x.device() || !a->IsContiguous())) return false;
    return true;
  }
  void operator()(Tensor x,Opt b,Opt t,Opt h,double s,Tensor w,Tensor i) const override {
    if(!supports(x,b,t,h,s,w,i)) throw std::invalid_argument("Invalid DeepSeek V4 routing inputs");
    auto launch=[&](auto tag) {
      using T=decltype(tag);
      v4_route::route<T><<<static_cast<unsigned>(std::min<size_t>((x.size(0)-1)/64+1,65535)),64,0,
                           static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const float*>(x.data()),b ? static_cast<const float*>(b->data()) : nullptr,
          t ? static_cast<const T*>(t->data()) : nullptr,h ? static_cast<const int64_t*>(h->data()) : nullptr,
          float(s),static_cast<float*>(w.data()),static_cast<int64_t*>(i.data()),
          x.size(0),x.size(1),w.size(1),h ? h->size(0) : 0);
    };
    if(t && t->dtype()==DataType::kInt32) launch(int32_t{});
    else launch(int64_t{});
    if(cudaGetLastError()!=cudaSuccess) throw std::runtime_error("DeepSeek V4 routing launch failed");
  }
};
} // namespace infini::ops
