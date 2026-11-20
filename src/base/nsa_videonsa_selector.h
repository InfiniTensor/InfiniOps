#pragma once
#include "common/op_utils/nsa_videonsa_common.h"
namespace infini::ops {
class NsaVideonsaSelector:public Operator<NsaVideonsaSelector>{
 public:
 static bool supports(const Tensor q,const Tensor k,const Tensor lse,int top,int block,float scale,Tensor out){
  if(!nsa_video_detail::base(q,k,top,block)||!nsa_video_detail::common(std::array<Tensor,4>{q,k,lse,out}))return false;
  return k.size(1)==(q.size(1)+block-1)/block&&std::isfinite(scale)&&lse.dtype()==DataType::kFloat32&&out.dtype()==DataType::kInt32&&
   lse.shape()==Tensor::Shape{q.size(0),q.size(2),q.size(1)}&&out.shape()==Tensor::Shape{q.size(0),q.size(1),k.size(2),size_t(top)};
 }
 NsaVideonsaSelector(const Tensor q,const Tensor k,const Tensor lse,int top,int block,float scale,Tensor out):meta_{q,k,lse,out},top_(top),block_(block),scale_(scale){
  if(!supports(q,k,lse,top,block,scale,out))throw std::invalid_argument("Invalid VideoNSA selector metadata");
 }
 virtual void operator()(const Tensor q,const Tensor k,const Tensor lse,int top,int block,float scale,Tensor out) const=0;
 protected:
 void Check(const Tensor q,const Tensor k,const Tensor lse,int top,int block,float scale,Tensor out) const{
  if(top!=top_||block!=block_||scale!=scale_||!supports(q,k,lse,top,block,scale,out)||!nsa_video_detail::same(meta_,std::array<Tensor,4>{q,k,lse,out}))throw std::invalid_argument("VideoNSA selector invocation differs from plan");
 }
 std::array<Tensor,4> meta_;int top_,block_;float scale_;
};
}
