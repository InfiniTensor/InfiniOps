#pragma once
#include "base/nsa_videonsa_prefill.h"
#include "base/nsa_videonsa_selector.h"
#include <cstdint>
#include <limits>
extern "C" {
size_t personal_nsa_selected_scratch_r50_matrix_clean2(int,int,int,int,int,int,int);
int personal_nsa_branch_r49_cdkv32(const void*,const void*,const void*,void*,float*,int,int,int,int,int,int,int,int,int,void*);
int personal_nsa_branch_r50_window_final(const void*,const void*,const void*,void*,float*,int,int,int,int,int,int,int,int,int,void*);
int personal_nsa_pool(const void*,const void*,void*,void*,int,int,int,int,int,void*);
int personal_nsa_prefill_selector(const void*,const void*,const float*,int32_t*,int,int,int,int,int,int,int,int,float,int,void*);
int personal_nsa_selected_r50_matrix_clean2(const void*,const void*,const void*,void*,const int32_t*,void*,int,int,int,int,int,int,float,int,void*);
int personal_nsa_gate_r51_final(const void*,const void*,const void*,const void*,const void*,const void*,void*,int64_t,int,int,void*);
}
namespace infini::ops {
namespace nsa_video_detail {
struct Plan {
 int b,t,hq,hkv,dtype,block,top,window,gate32;size_t workspace{};std::array<size_t,8> offsets{};
 Plan(const Tensor q,const Tensor k,const Tensor gate,int tp,int bs,int win):b(q.size(0)),t(q.size(1)),hq(q.size(2)),hkv(k.size(2)),dtype(q.dtype()==DataType::kBFloat16),block(bs),top(tp),window(win),gate32(gate.dtype()==DataType::kFloat32){
  const size_t compressed=size_t(b)*((t+bs-1)/bs)*hkv*128*2,output=size_t(b)*t*hq*128*2;
  offsets[0]=reserve(compressed);offsets[1]=reserve(compressed);offsets[2]=reserve(size_t(b)*t*hq*4);offsets[3]=reserve(size_t(b)*t*hkv*top*4);
  for(int i=4;i<7;++i)offsets[i]=reserve(output);
  if(hq/hkv==7&&bs==64&&t>=1024)offsets[7]=reserve(size_t(b)*hkv*((t+63)/64)*128*64*2);
  auto extra=personal_nsa_selected_scratch_r50_matrix_clean2(b,t,hq,hkv,top,bs,dtype);if(extra)offsets[7]=reserve(extra);
 }
 size_t reserve(size_t bytes){if(bytes>std::numeric_limits<size_t>::max()-workspace-255)throw std::overflow_error("NSA workspace overflow");auto at=workspace;workspace=(workspace+bytes+255)&~size_t(255);return at;}
};
inline void launch_check(int code){if(code)throw std::runtime_error("Hygon VideoNSA launch failed: "+std::to_string(code));}
}
template<> class Operator<NsaVideonsaPrefill,Device::Type::kHygon>:public NsaVideonsaPrefill{
 nsa_video_detail::Plan plan_;
 public:
 static bool supports(const Tensor q,const Tensor k,const Tensor v,const Tensor gc,const Tensor gs,const Tensor gw,int top,int block,int window,Tensor out){return q.device().type()==Device::Type::kHygon&&NsaVideonsaPrefill::supports(q,k,v,gc,gs,gw,top,block,window,out);}
 Operator(const Tensor q,const Tensor k,const Tensor v,const Tensor gc,const Tensor gs,const Tensor gw,int top,int block,int window,Tensor out):NsaVideonsaPrefill(q,k,v,gc,gs,gw,top,block,window,out),plan_(q,k,gc,top,block,window){}
 size_t workspace_size_in_bytes() const override{return plan_.workspace;}
 bool capture_safe() const override{return true;}
 void operator()(const Tensor q,const Tensor k,const Tensor v,const Tensor gc,const Tensor gs,const Tensor gw,int top,int block,int window,Tensor out) const override{
  Check(q,k,v,gc,gs,gw,top,block,window,out);
  if(!workspace_||workspace_size_in_bytes_<plan_.workspace||reinterpret_cast<uintptr_t>(workspace_)%256)throw std::invalid_argument("VideoNSA prefill requires sufficient 256-aligned workspace");
  const auto& m=plan_;auto ptr=[&](int i){return static_cast<void*>(static_cast<unsigned char*>(workspace_)+m.offsets[i]);};const int blocks=(m.t+m.block-1)/m.block;constexpr float scale=0x1.6a09e6p-4f;
  using nsa_video_detail::launch_check;
  launch_check(personal_nsa_pool(k.data(),v.data(),ptr(0),ptr(1),m.b,m.t,m.hkv,m.block,m.dtype,stream_));
  launch_check(personal_nsa_branch_r49_cdkv32(q.data(),ptr(0),ptr(1),ptr(4),static_cast<float*>(ptr(2)),m.b,m.t,blocks,m.hq,m.hkv,m.block,m.window,0,m.dtype,stream_));
  launch_check(personal_nsa_prefill_selector(q.data(),ptr(0),static_cast<const float*>(ptr(2)),static_cast<int32_t*>(ptr(3)),m.b,m.t,m.hq,m.hkv,128,blocks,m.block,m.top,scale,m.dtype,stream_));
  void* packed=((m.hq/m.hkv==7&&m.block==64&&m.t>=1024)||personal_nsa_selected_scratch_r50_matrix_clean2(m.b,m.t,m.hq,m.hkv,m.top,m.block,m.dtype))?ptr(7):nullptr;
  launch_check(personal_nsa_selected_r50_matrix_clean2(q.data(),k.data(),v.data(),packed,static_cast<const int32_t*>(ptr(3)),ptr(5),m.b,m.t,m.hq,m.hkv,m.top,m.block,scale,m.dtype,stream_));
  launch_check(personal_nsa_branch_r50_window_final(q.data(),k.data(),v.data(),ptr(6),nullptr,m.b,m.t,m.t,m.hq,m.hkv,m.block,m.window,1,m.dtype,stream_));
  launch_check(personal_nsa_gate_r51_final(ptr(4),ptr(5),ptr(6),gc.data(),gs.data(),gw.data(),out.data(),int64_t(m.b)*m.t*m.hq*128,m.dtype,m.gate32,stream_));
 }
};
}
