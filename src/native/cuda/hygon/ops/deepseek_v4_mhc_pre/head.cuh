#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
// Match the reference's 512-column head projection tiles and warp reductions.
// This is separate from the split-K decoder mHC projection.
namespace infini::ops::v4_mhc_head {
static __device__ float wave_sum(float x) {
 for(unsigned offset=8;offset;offset/=2)x+=__shfl_xor(x,offset,64);
 x+=__shfl_xor(x,16,64);x+=__shfl_xor(x,32,64);return x;
}
static __global__ void head(const __nv_bfloat16* x,const float* weight,const float* scale,const float* bias,
                           __nv_bfloat16* out,size_t hidden,float eps,float hc_eps) {
 __shared__ float squares[4],pre[4];
 const size_t row=blockIdx.x,width=hidden*4;
 const unsigned tid=threadIdx.x,lane=tid%64,col=tid/64;
 float dot=0,sum_sq=0;
 for(size_t tile=0;tile<width;tile+=512) {
  float a=__bfloat162float(x[row*width+tile+tid*2]);
  float b=__bfloat162float(x[row*width+tile+tid*2+1]);
  float sq=wave_sum(fmaf(a,a,b*b));
  if(!lane)squares[col]=sq;
  const size_t k=tile+lane*8;
  float chunk=__bfloat162float(x[row*width+k+1])*weight[col*width+k+1];
  chunk=fmaf(__bfloat162float(x[row*width+k]),weight[col*width+k],chunk);
  for(unsigned j=2;j<8;++j)
   chunk=fmaf(__bfloat162float(x[row*width+k+j]),weight[col*width+k+j],chunk);
  dot+=wave_sum(chunk);
  __syncthreads();
  sum_sq+=(squares[0]+squares[2])+(squares[1]+squares[3]);
  __syncthreads();
 }
 if(!lane) {
  const float rms=rsqrtf(fmaf(sum_sq,1.f/float(width),eps));
  const float value=fmaf(dot*rms,scale[0],bias[col]);
  pre[col]=1.f/(1.f+__builtin_amdgcn_exp2f(-value*1.4426950408889634f))+hc_eps;
 }
 __syncthreads();
 for(size_t i=tid;i<hidden;i+=256) {
  float terms[4];
  for(unsigned h=0;h<4;++h)terms[h]=__fmul_rn(__bfloat162float(x[(row*4+h)*hidden+i]),pre[h]);
  const float sum=__fadd_rn(__fadd_rn(terms[0],terms[2]),__fadd_rn(terms[1],terms[3]));
  out[row*hidden+i]=__float2bfloat16_rn(sum);
 }
}
}
