#pragma once
#include "operator.h"
#include <array>
#include <stdexcept>
namespace infini::ops {
class NsaCompressPagedCache : public Operator<NsaCompressPagedCache> {
 public:
  static bool supports(const Tensor kc,const Tensor vc,const Tensor bt,const Tensor lens,
      int block,bool tail,Tensor ko,Tensor vo) {
    if(kc.ndim()!=4 || vc.shape()!=kc.shape() || ko.ndim()!=3 || vo.shape()!=ko.shape() ||
       bt.ndim()!=2 || lens.ndim()!=1 || lens.size(0)!=bt.size(0) ||
       (kc.dtype()!=DataType::kFloat16 && kc.dtype()!=DataType::kBFloat16) ||
       vc.dtype()!=kc.dtype() || ko.dtype()!=kc.dtype() || vo.dtype()!=kc.dtype() ||
       lens.dtype()!=bt.dtype() || (bt.dtype()!=DataType::kInt32 && bt.dtype()!=DataType::kInt64 && bt.dtype()!=DataType::kUInt32) ||
       !kc.size(0) || !kc.size(1) || !kc.size(2) || kc.size(3)!=128 || block<=0 ||
       kc.size(2)%size_t(block)!=0 || bt.size(1)==0 ||
       ko.size(0)!=kc.size(0)*(kc.size(2)/size_t(block)) || ko.size(1)!=kc.size(1) || ko.size(2)!=128 ||
       kc.stride(3)!=1 || vc.stride(3)!=1 || ko.stride(2)!=1 || vo.stride(2)!=1 || bt.stride(1)!=1 || lens.stride(0)!=1) return false;
    for(const auto& t:{kc,vc,bt,lens,ko,vo}) {
      if(t.device()!=kc.device())return false;
      for(auto stride:t.strides())if(stride<0)return false;
    }
    (void)tail; return true;
  }
  NsaCompressPagedCache(const Tensor kc,const Tensor vc,const Tensor bt,const Tensor lens,
      int block,bool tail,Tensor ko,Tensor vo):metadata_{kc,vc,bt,lens,ko,vo},block_{block},tail_{tail} {
    if(!supports(kc,vc,bt,lens,block,tail,ko,vo)) throw std::invalid_argument("Invalid NSA compressed paged cache metadata");
  }
  virtual void operator()(const Tensor kc,const Tensor vc,const Tensor bt,const Tensor lens,
      int block,bool tail,Tensor ko,Tensor vo) const=0;
 protected:
  void Check(const Tensor kc,const Tensor vc,const Tensor bt,const Tensor lens,
      int block,bool tail,Tensor ko,Tensor vo) const {
    if(block!=block_ || tail!=tail_ || !supports(kc,vc,bt,lens,block,tail,ko,vo))throw std::invalid_argument("NSA compressed cache invocation differs from plan");
    const std::array<Tensor,6> values{kc,vc,bt,lens,ko,vo};
    for(size_t i=0;i<values.size();++i)if(!std::equal_to<Tensor>{}(values[i],metadata_[i]))throw std::invalid_argument("NSA compressed cache tensor metadata differs from plan");
  }
  std::array<Tensor,6> metadata_;
  int block_; bool tail_;
};
}
