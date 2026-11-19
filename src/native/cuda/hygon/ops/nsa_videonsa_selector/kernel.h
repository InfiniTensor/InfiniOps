#pragma once
#include "base/nsa_videonsa_selector.h"
#include <cstdint>
extern "C" int personal_nsa_prefill_selector(const void*,const void*,const float*,int32_t*,int,int,int,int,int,int,int,int,float,int,void*);
namespace infini::ops {
template<> class Operator<NsaVideonsaSelector,Device::Type::kHygon>:public NsaVideonsaSelector{
 public:
 using NsaVideonsaSelector::NsaVideonsaSelector;
 static bool supports(const Tensor q,const Tensor k,const Tensor lse,int top,int block,float scale,Tensor out){return q.device().type()==Device::Type::kHygon&&NsaVideonsaSelector::supports(q,k,lse,top,block,scale,out);}
 bool capture_safe() const override{return true;}
 void operator()(const Tensor q,const Tensor k,const Tensor lse,int top,int block,float scale,Tensor out) const override{
  Check(q,k,lse,top,block,scale,out);
  const auto code=personal_nsa_prefill_selector(q.data(),k.data(),static_cast<const float*>(lse.data()),static_cast<int32_t*>(out.data()),q.size(0),q.size(1),q.size(2),k.size(2),128,k.size(1),block,top,scale,q.dtype()==DataType::kBFloat16,stream_);
  if(code)throw std::runtime_error("Hygon VideoNSA selector launch failed");
 }
};
}
