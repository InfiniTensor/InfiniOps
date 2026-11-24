#pragma once
#include "operator.h"
#include <array>
#include <cmath>
#include <stdexcept>
namespace infini::ops::nsa_video_detail {
inline bool base(const Tensor q,const Tensor k,int top,int block){
 if(q.ndim()!=4||k.ndim()!=4||!q.IsContiguous()||!k.IsContiguous()||
  (q.dtype()!=DataType::kFloat16&&q.dtype()!=DataType::kBFloat16)||k.dtype()!=q.dtype())return false;
 const auto b=q.size(0),t=q.size(1),hq=q.size(2),hkv=k.size(2);
 return b>=1&&b<=8&&t>=1&&t<=8192&&hq>=1&&hq<=1024&&hkv>=1&&hkv<=32&&hq%hkv==0&&hq/hkv<=32&&
  q.size(3)==128&&k.size(0)==b&&k.size(3)==128&&block>=2&&block<=128&&(block&(block-1))==0&&top>=1&&top<=32&&top<=block/2;
}
template<size_t N> inline bool common(const std::array<Tensor,N>& ts){
 for(const auto& t:ts)if(t.device()!=ts[0].device()||!t.IsContiguous())return false;
 return true;
}
template<size_t N> inline bool same(const std::array<Tensor,N>& a,const std::array<Tensor,N>& b){
 for(size_t i=0;i<N;++i)if(!std::equal_to<Tensor>{}(a[i],b[i]))return false;
 return true;
}
}
