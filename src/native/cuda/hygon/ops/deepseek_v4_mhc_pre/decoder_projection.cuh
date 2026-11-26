#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace infini::ops::v4_mhc_opt {
using Float2 = float __attribute__((ext_vector_type(2)));
using Float4 = float __attribute__((ext_vector_type(4)));
constexpr unsigned Splits=32, Outputs=24, Threads=128;

// Adjacent BF16/FP32 loads preserve the original MMAC4/MMAC8 instruction order.
// Only the first output tile needs the squared norm; keep that branch outside K.
template<bool Pack8, bool Squares, bool VectorLoad>
static __device__ __forceinline__ void split_body(
    const __nv_bfloat16* x,const float* w,float* dots,float* squares,
    size_t rows,size_t width) {
  const unsigned lane=threadIdx.x,group=lane/16;
  const size_t row=blockIdx.x*16+lane%16,col=blockIdx.y*16+lane%16;
  const size_t start=blockIdx.z*(width/Splits);
  Float4 dot={}; float square=0.0f;
  for(size_t tile=0;tile<width/Splits;tile+=64) {
    for(unsigned j=0;j<16;j+=2) {
      const size_t k=start+tile+(j/8*4+group)*8+j%8;
      float a0,a1,b0,b1;
      if constexpr(VectorLoad) {
        const unsigned raw=row<rows ? *reinterpret_cast<const unsigned*>(x+row*width+k) : 0u;
        a0=__uint_as_float(raw<<16); a1=__uint_as_float(raw&0xffff0000u);
        const Float2 b=col<Outputs ? *reinterpret_cast<const Float2*>(w+col*width+k) : Float2{};
        b0=b[0]; b1=b[1];
      } else {
        a0=row<rows?__bfloat162float(x[row*width+k]):0.0f;
        a1=row<rows?__bfloat162float(x[row*width+k+1]):0.0f;
        b0=col<Outputs?w[col*width+k]:0.0f;
        b1=col<Outputs?w[col*width+k+1]:0.0f;
      }
      if constexpr(Squares) { square=fmaf(a0,a0,square);square=fmaf(a1,a1,square); }
      if constexpr(Pack8) dot=__builtin_hcu_mmac_16x16x8_f32(Float2{a0,a1},Float2{b0,b1},dot);
      else {
        dot=__builtin_hcu_mmac_16x16x4_f32(a0,b0,dot);
        dot=__builtin_hcu_mmac_16x16x4_f32(a1,b1,dot);
      }
    }
  }
  if constexpr(Squares) {
    const float pair=square+__shfl_xor(square,32,64);
    const float total=pair+__shfl_xor(pair,16,64);
    if(row<rows && group==0)squares[blockIdx.z*rows+row]=total;
  }
  if(row>=rows)return;
  for(unsigned i=0;i<4;++i) {
    const size_t column=blockIdx.y*16+group+i*4;
    if(column<Outputs)dots[(blockIdx.z*rows+row)*Outputs+column]=dot[i];
  }
}
template<bool Pack8,bool VectorLoad=true>
static __global__ void project_split(const __nv_bfloat16* x,const float* w,
                                    float* dots,float* squares,size_t rows,size_t width) {
  if(blockIdx.y==0)split_body<Pack8,true,VectorLoad>(x,w,dots,squares,rows,width);
  else split_body<Pack8,false,VectorLoad>(x,w,dots,squares,rows,width);
}

// Larger prefill fallback: reuse each input load and squared norm across four
// columns, retaining the original per-column K order and 128-thread reduction.
static __global__ void project_tiled(const __nv_bfloat16* x,const float* w,float* values,
                                    size_t rows,size_t width,size_t outputs,float eps) {
  __shared__ float reduce[5][Threads];
  const unsigned lane=threadIdx.x;
  const size_t column=blockIdx.y*4;
  for(size_t row=blockIdx.x;row<rows;row+=gridDim.x) {
    float dots[4]={},square=0.0f;
    for(size_t k=lane;k<width;k+=Threads) {
      const float a=__bfloat162float(x[row*width+k]);
#pragma unroll
      for(unsigned j=0;j<4;++j)if(column+j<outputs)dots[j]+=a*w[(column+j)*width+k];
      square+=a*a;
    }
#pragma unroll
    for(unsigned j=0;j<4;++j)reduce[j][lane]=dots[j];
    reduce[4][lane]=square;
    __syncthreads();
    for(unsigned step=Threads/2;step;step/=2) {
      if(lane<step) {
#pragma unroll
        for(unsigned j=0;j<5;++j)reduce[j][lane]+=reduce[j][lane+step];
      }
      __syncthreads();
    }
    if(lane==0) {
      const float inverse=rsqrtf(reduce[4][0]/float(width)+eps);
#pragma unroll
      for(unsigned j=0;j<4;++j)if(column+j<outputs)values[row*outputs+column+j]=reduce[j][0]*inverse;
    }
    __syncthreads();
  }
}
} // namespace infini::ops::v4_mhc_opt
