#pragma once
#include "base/nsa_compress_paged_cache.h"
#include <array>
#include <cmath>
namespace infini::ops {
class NsaPagedAttention : public Operator<NsaPagedAttention> {
 public:
  static bool supports(const Tensor q,const Tensor km,const Tensor vm,const Tensor kc,const Tensor vc,const Tensor bt,const Tensor lens,const Tensor gates,
      float scale,int block,int window,int selected,Tensor out) {
    if(!NsaCompressPagedCache::supports(kc,vc,bt,lens,block,false,km,vm) ||
       q.ndim()!=3 || q.size(0)!=bt.size(0) || !q.size(1) || q.size(1)%kc.size(1)!=0 || q.size(2)!=128 ||
       q.dtype()!=kc.dtype() || out.shape()!=q.shape() || out.dtype()!=q.dtype() || q.stride(2)!=1 || out.stride(2)!=1 ||
       gates.ndim()!=3 || gates.shape()!=Tensor::Shape{q.size(0),3,q.size(1)} ||
       (gates.dtype()!=q.dtype() && gates.dtype()!=DataType::kFloat32) || !std::isfinite(scale) || window<0 || selected<=0)return false;
    for(const auto& t:{q,out,gates}) {
      if(t.device()!=kc.device())return false;
      for(auto stride:t.strides())if(stride<0)return false;
    }
    return true;
  }
  NsaPagedAttention(const Tensor q,const Tensor km,const Tensor vm,const Tensor kc,const Tensor vc,const Tensor bt,const Tensor lens,const Tensor gates,
      float scale,int block,int window,int selected,Tensor out):metadata_{q,km,vm,kc,vc,bt,lens,gates,out},scale_{scale},block_{block},window_{window},selected_{selected} {
    if(!supports(q,km,vm,kc,vc,bt,lens,gates,scale,block,window,selected,out))throw std::invalid_argument("Invalid NSA paged attention metadata");
  }
  virtual void operator()(const Tensor q,const Tensor km,const Tensor vm,const Tensor kc,const Tensor vc,const Tensor bt,const Tensor lens,const Tensor gates,
      float scale,int block,int window,int selected,Tensor out) const=0;
 protected:
  void Check(const Tensor q,const Tensor km,const Tensor vm,const Tensor kc,const Tensor vc,const Tensor bt,const Tensor lens,const Tensor gates,
      float scale,int block,int window,int selected,Tensor out) const {
    if(scale!=scale_ || block!=block_ || window!=window_ || selected!=selected_ || !supports(q,km,vm,kc,vc,bt,lens,gates,scale,block,window,selected,out))throw std::invalid_argument("NSA paged attention invocation differs from plan");
    const std::array<Tensor,9> values{q,km,vm,kc,vc,bt,lens,gates,out};
    for(size_t i=0;i<values.size();++i)if(!std::equal_to<Tensor>{}(values[i],metadata_[i]))throw std::invalid_argument("NSA paged attention tensor metadata differs from plan");
  }
  std::array<Tensor,9> metadata_;float scale_;int block_,window_,selected_;
};
}
