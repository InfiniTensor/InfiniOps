#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace infini::ops::v4_mhc_opt {
static __device__ __forceinline__ float ordered_splits(const float* values,size_t stride) {
  float partials[4]={};
  for(unsigned split=0;split<32;split+=4)
    for(unsigned group=0;group<4;++group)partials[group]+=values[(split+group)*stride];
  return ((partials[0]+partials[1])+partials[2])+partials[3];
}
static __device__ __forceinline__ float ordered_pairs(float a,float b,float c,float d) {
  return (a+b)+(c+d);
}
static __device__ __forceinline__ float wave_value(float value,unsigned source) {
  return __shfl(value,source,64);
}
template<bool Split>
static __device__ __forceinline__ float logistic(float value) {
  return 1.0f/(1.0f+(Split?__builtin_amdgcn_exp2f(-value*1.4426950408889634f):expf(-value)));
}

// Lanes 0..3 produce pre, 4..7 post, and 8..23 the 4x4 matrix. Every matrix
// element follows the scalar reference's exact FP32 operation/reduction order.
template<bool Split>
static __global__ void mix_parallel(const __nv_bfloat16* residual,const float* values,
    const float* scales,const float* bias,__nv_bfloat16* collapsed,float* post,
    float* combination,size_t rows,size_t hidden,float eps,int64_t iterations,
    const float* squares,float norm_eps) {
  __shared__ float pre[4];
  const unsigned tid=threadIdx.x;
  for(size_t row=blockIdx.x;row<rows;row+=gridDim.x) {
    float value=0.0f;
    if(tid<24) {
      if constexpr(Split) {
        const float inverse=rsqrtf(ordered_splits(squares+row,rows)/float(4*hidden)+norm_eps);
        value=ordered_splits(values+row*24+tid,rows*24)*inverse;
      } else value=values[row*24+tid];
    }
    if(tid<4)pre[tid]=logistic<Split>(value*scales[0]+bias[tid])+eps;
    if(tid>=4 && tid<8)post[row*4+tid-4]=2*logistic<Split>(value*scales[1]+bias[tid]);
    if(tid>=8 && tid<24) {
      const unsigned i=(tid-8)/4,j=(tid-8)%4,base=8+i*4;
      float matrix=value*scales[2]+bias[tid];
      const float m0=wave_value(matrix,base),m1=wave_value(matrix,base+1);
      const float m2=wave_value(matrix,base+2),m3=wave_value(matrix,base+3);
      float maximum=-INFINITY;
      maximum=fmaxf(maximum,m0);maximum=fmaxf(maximum,m1);
      maximum=fmaxf(maximum,m2);maximum=fmaxf(maximum,m3);
      matrix=expf(matrix-maximum);
      const float e0=wave_value(matrix,base),e1=wave_value(matrix,base+1);
      const float e2=wave_value(matrix,base+2),e3=wave_value(matrix,base+3);
      float total;
      if constexpr(Split)total=ordered_pairs(e0,e1,e2,e3);
      else {total=0.0f;total+=e0;total+=e1;total+=e2;total+=e3;}
      matrix=matrix/total+eps;
      for(int64_t iteration=0;iteration<iterations;++iteration) {
        if(iteration) {
          const float a=wave_value(matrix,base),b=wave_value(matrix,base+1);
          const float c=wave_value(matrix,base+2),d=wave_value(matrix,base+3);
          if constexpr(Split)total=ordered_pairs(a,b,c,d)+eps;
          else {total=eps;total+=a;total+=b;total+=c;total+=d;}
          matrix/=total;
        }
        const float a=wave_value(matrix,8+j),b=wave_value(matrix,12+j);
        const float c=wave_value(matrix,16+j),d=wave_value(matrix,20+j);
        if constexpr(Split) {
          const float even=ordered_pairs(a,d,c,b)+eps;
          const float odd=ordered_pairs(b,a,d,c)+eps;
          matrix/=(i%2?odd:even);
        } else {
          total=eps;total+=a;total+=b;total+=c;total+=d;matrix/=total;
        }
      }
      combination[row*16+tid-8]=matrix;
    }
    __syncthreads();
    for(size_t col=tid;col<hidden;col+=blockDim.x) {
      if constexpr(Split) {
        const float a=__bfloat162float(residual[row*4*hidden+col]);
        const float b=__bfloat162float(residual[(row*4+1)*hidden+col]);
        const float c=__bfloat162float(residual[(row*4+2)*hidden+col]);
        const float d=__bfloat162float(residual[(row*4+3)*hidden+col]);
        const float left=fmaf(a,pre[0],b*pre[1]);
        const float right=fmaf(c,pre[2],d*pre[3]);
        collapsed[row*hidden+col]=__float2bfloat16_rz(left+right);
      } else {
        float sum=0.0f;
        for(unsigned stream=0;stream<4;++stream)
          sum+=__bfloat162float(residual[(row*4+stream)*hidden+col])*pre[stream];
        collapsed[row*hidden+col]=__float2bfloat16(sum);
      }
    }
    __syncthreads();
  }
}
} // namespace infini::ops::v4_mhc_opt
