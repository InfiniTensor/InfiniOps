#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace infini::ops::v4_mhc_post_opt {
// Reuse the four residual inputs for the four output streams. Do not reassociate
// the residual-first FMA chain or change the final round-toward-zero conversion.
static __global__ void mix_reuse(const __nv_bfloat16* activation,const __nv_bfloat16* residual,
    const float* post,const float* matrix,__nv_bfloat16* out,size_t rows,size_t hidden) {
  const size_t count=rows*hidden;
  for(size_t index=blockIdx.x*blockDim.x+threadIdx.x;index<count;index+=gridDim.x*blockDim.x) {
    const size_t row=index/hidden,col=index%hidden;
    const float r0=__bfloat162float(residual[row*4*hidden+col]);
    const float r1=__bfloat162float(residual[(row*4+1)*hidden+col]);
    const float r2=__bfloat162float(residual[(row*4+2)*hidden+col]);
    const float r3=__bfloat162float(residual[(row*4+3)*hidden+col]);
    const float x=__bfloat162float(activation[row*hidden+col]);
#pragma unroll
    for(unsigned stream=0;stream<4;++stream) {
      float value=matrix[row*16+stream]*r0;
      value=fmaf(post[row*4+stream],x,value);
      value=fmaf(matrix[row*16+4+stream],r1,value);
      value=fmaf(matrix[row*16+8+stream],r2,value);
      value=fmaf(matrix[row*16+12+stream],r3,value);
      out[(row*4+stream)*hidden+col]=__float2bfloat16_rz(value);
    }
  }
}
} // namespace infini::ops::v4_mhc_post_opt
