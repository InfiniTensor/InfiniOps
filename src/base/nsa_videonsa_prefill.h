#pragma once
#include "nsa_videonsa_common.h"
namespace infini::ops {
class NsaVideonsaPrefill:public Operator<NsaVideonsaPrefill>{
 public:
 static bool supports(const Tensor q,const Tensor k,const Tensor v,const Tensor gc,const Tensor gs,const Tensor gw,int top,int block,int window,Tensor out){
  if(!nsa_video_detail::base(q,k,top,block)||!nsa_video_detail::common(std::array<Tensor,7>{q,k,v,gc,gs,gw,out}))return false;
  return k.size(1)==q.size(1)&&v.shape()==k.shape()&&v.dtype()==q.dtype()&&out.shape()==q.shape()&&window>=0&&
   (gc.dtype()==q.dtype()||gc.dtype()==DataType::kFloat32)&&gs.dtype()==gc.dtype()&&gw.dtype()==gc.dtype()&&out.dtype()==gc.dtype()&&
   gc.shape()==Tensor::Shape{q.size(0),q.size(1),q.size(2)}&&gs.shape()==gc.shape()&&gw.shape()==gc.shape();
 }
 NsaVideonsaPrefill(const Tensor q,const Tensor k,const Tensor v,const Tensor gc,const Tensor gs,const Tensor gw,int top,int block,int window,Tensor out):meta_{q,k,v,gc,gs,gw,out},top_(top),block_(block),window_(window){
  if(!supports(q,k,v,gc,gs,gw,top,block,window,out))throw std::invalid_argument("Invalid VideoNSA prefill metadata");
 }
 virtual void operator()(const Tensor q,const Tensor k,const Tensor v,const Tensor gc,const Tensor gs,const Tensor gw,int top,int block,int window,Tensor out) const=0;
 protected:
 void Check(const Tensor q,const Tensor k,const Tensor v,const Tensor gc,const Tensor gs,const Tensor gw,int top,int block,int window,Tensor out) const{
  if(top!=top_||block!=block_||window!=window_||!supports(q,k,v,gc,gs,gw,top,block,window,out)||!nsa_video_detail::same(meta_,std::array<Tensor,7>{q,k,v,gc,gs,gw,out}))throw std::invalid_argument("VideoNSA prefill invocation differs from plan");
 }
 std::array<Tensor,7> meta_;int top_,block_,window_;
};
}
