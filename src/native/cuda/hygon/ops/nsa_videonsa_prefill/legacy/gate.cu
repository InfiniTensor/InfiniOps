#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#include <type_traits>
#include <mma.h>
__device__ __forceinline__ unsigned int r51_keep_bits(unsigned int bits) {
#if defined(__AMDGCN__)
    asm volatile("" : "+v"(bits));
#else
    asm volatile("" : "+r"(bits));
#endif
    return bits;
}
__device__ __forceinline__ float r51_half_round(float x) {
    const unsigned int bits=r51_keep_bits(__half_as_ushort(__float2half_rn(x)));
    return __half2float(__ushort_as_half((unsigned short)bits));
}
template<class G> __device__ __forceinline__ float r51_gate_round(float x) {
    if constexpr(std::is_same<G,half>::value)return r51_half_round(x);
    else if constexpr(std::is_same<G,__nv_bfloat16>::value) {
        const unsigned int bits=r51_keep_bits(__bfloat16_as_ushort(__float2bfloat16_rn(x)));
        return __bfloat162float(__ushort_as_bfloat16((unsigned short)bits));
    } else {
        return __uint_as_float(r51_keep_bits(__float_as_uint(x)));
    }
}

template<class T,class G> static __device__ __forceinline__ G r51_fuse(T oc,T os,T ow,G gc,G gs,G gw){
    const float pc=r51_gate_round<G>(__fmul_rn(float(oc),float(gc)));
    const float ps=r51_gate_round<G>(__fmul_rn(float(os),float(gs)));
    const float pw=r51_gate_round<G>(__fmul_rn(float(ow),float(gw)));
    const float partial=r51_gate_round<G>(__fadd_rn(pc,ps));
    return G(__fadd_rn(partial,pw));
}

extern "C" int personal_nsa_gate_r49_reg(const void*,const void*,const void*,const void*,const void*,const void*,void*,int64_t,int,int,void*);
extern "C" size_t personal_nsa_selected_scratch_r50_matrix_clean2(int,int,int,int,int,int,int);

template<class T,class G,int N> __global__ void nsaGateR51FinalCuda(const T* oc,const T* os,const T* ow,const G* gc,const G* gs,const G* gw,G* out,int64_t elements){
    const int64_t i=(int64_t(blockIdx.x)*blockDim.x+threadIdx.x)*N;
    if(i>=elements)return;
    using Load=typename std::conditional<N==2,uint32_t,typename std::conditional<N==4,uint64_t,uint4>::type>::type;
    T c[N],s[N],w[N];
    Load cv=*reinterpret_cast<const Load*>(oc+i),sv=*reinterpret_cast<const Load*>(os+i),wv=*reinterpret_cast<const Load*>(ow+i);
    __builtin_memcpy(c,&cv,sizeof(c));__builtin_memcpy(s,&sv,sizeof(s));__builtin_memcpy(w,&wv,sizeof(w));
    const int64_t row=i/128;const G a=gc[row],b=gs[row],d=gw[row];G values[N];
    #pragma unroll
    for(int j=0;j<N;++j)values[j]=r51_fuse<T,G>(c[j],s[j],w[j],a,b,d);
    if constexpr(sizeof(G)*N==4){uint32_t bits;__builtin_memcpy(&bits,values,4);*reinterpret_cast<uint32_t*>(out+i)=bits;}
    else if constexpr(sizeof(G)*N==8){uint64_t bits;__builtin_memcpy(&bits,values,8);*reinterpret_cast<uint64_t*>(out+i)=bits;}
    else if constexpr(sizeof(G)*N==16){uint4 bits;__builtin_memcpy(&bits,values,16);*reinterpret_cast<uint4*>(out+i)=bits;}
    else {uint4 bits[2];__builtin_memcpy(bits,values,32);reinterpret_cast<uint4*>(out+i)[0]=bits[0];reinterpret_cast<uint4*>(out+i)[1]=bits[1];}
}
extern "C" int personal_nsa_gate_r51_final(const void* oc,const void* os,const void* ow,const void* gc,const void* gs,const void* gw,void* out,int64_t n,int dtype,int gate32,void* stream){
    const uintptr_t alignment=8*2,out_alignment=gate32?(4*8>16?16:4*8):8*2;
    if(n<1048576||n%8!=0||(reinterpret_cast<uintptr_t>(oc)%alignment)||(reinterpret_cast<uintptr_t>(os)%alignment)||(reinterpret_cast<uintptr_t>(ow)%alignment)||(reinterpret_cast<uintptr_t>(out)%out_alignment))return personal_nsa_gate_r49_reg(oc,os,ow,gc,gs,gw,out,n,dtype,gate32,stream);
    const dim3 grid((n+256*8-1)/(256*8));
    if(dtype==0&&gate32==0)nsaGateR51FinalCuda<half,half,8><<<grid,256,0,(cudaStream_t)stream>>>((const half*)oc,(const half*)os,(const half*)ow,(const half*)gc,(const half*)gs,(const half*)gw,(half*)out,n);
    else if(dtype==1&&gate32==0)nsaGateR51FinalCuda<__nv_bfloat16,__nv_bfloat16,8><<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)oc,(const __nv_bfloat16*)os,(const __nv_bfloat16*)ow,(const __nv_bfloat16*)gc,(const __nv_bfloat16*)gs,(const __nv_bfloat16*)gw,(__nv_bfloat16*)out,n);
    else if(dtype==0)nsaGateR51FinalCuda<half,float,8><<<grid,256,0,(cudaStream_t)stream>>>((const half*)oc,(const half*)os,(const half*)ow,(const float*)gc,(const float*)gs,(const float*)gw,(float*)out,n);
    else nsaGateR51FinalCuda<__nv_bfloat16,float,8><<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)oc,(const __nv_bfloat16*)os,(const __nv_bfloat16*)ow,(const float*)gc,(const float*)gs,(const float*)gw,(float*)out,n);
    return int(cudaGetLastError());
}
extern "C" int personal_nsa_resources_r51_final(int* values,int gate32){
    cudaFuncAttributes a;auto e=gate32?cudaFuncGetAttributes(&a,nsaGateR51FinalCuda<__nv_bfloat16,float,8>):cudaFuncGetAttributes(&a,nsaGateR51FinalCuda<__nv_bfloat16,__nv_bfloat16,8>);if(e!=cudaSuccess)return int(e);int blocks=0;
    e=gate32?cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,nsaGateR51FinalCuda<__nv_bfloat16,float,8>,256,0):cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,nsaGateR51FinalCuda<__nv_bfloat16,__nv_bfloat16,8>,256,0);values[0]=a.numRegs;values[1]=a.sharedSizeBytes;values[2]=a.localSizeBytes;values[3]=blocks;return int(e);
}
