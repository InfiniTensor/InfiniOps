
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>
extern "C" int personal_nsa_selected_packed_r46_bf16u8(const void*,const void*,const void*,void*,const int32_t*,void*,int,int,int,int,int,int,float,int,void*);
template<class T> __global__ __launch_bounds__(256)
void nsaSelectedR48Exact16Cuda(const T* q,const T* k,const T* v,const int32_t* ids,T* out,
 int tokens,int hq,int hkv,int width,int bs,float scale) {
    using namespace nvcuda;
    constexpr int DS=136,NS=72;
    const int tid=threadIdx.x,warp=tid/warpSize,lane=tid%32;
    const int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ float rounded_q[128][17];
    __shared__ __align__(32) float scores[16*NS];
    __shared__ float maximum[16],denom[16],alpha[16];
    for(int i=tid;i<16*128;i+=256) {
        const int head=i/128,d=i%128;
        rounded_q[d][head]=float(T(float(q[((int64_t(batch)*tokens+query)*hq+kv*16+head)*128+d])*scale));
    }
    if(tid<16){maximum[tid]=-INFINITY;denom[tid]=0.f;}
    const int pv_lane=tid%64,pv_head=(tid/64)*4;
    float acc[8]={};
    __syncthreads();
    for(int slot=0;slot<width;++slot) {
        const int id=ids[io+slot];
        if(id<0||int64_t(id)*64>query)continue;
        const int start=id*64,valid=min(64,query+1-start);
        for(int task=tid;task<16*64;task+=256) {
            const int lh=task%16,j=task/16;
            float dot=-INFINITY;
            if(j<valid) {
                const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                float parts[32]={};
                #pragma unroll
                for(int i=0;i<4;++i) {
                    #pragma unroll
                    for(int l=0;l<32;++l)parts[l]=fmaf(rounded_q[l+i*32][lh],float(k[kb+l+i*32]),parts[l]);
                }
                #pragma unroll
                for(int delta=16;delta;delta/=2) {
                    #pragma unroll
                    for(int l=0;l<delta;++l)parts[l]+=parts[l+delta];
                }
                dot=parts[0];
            }
            scores[lh*NS+j]=dot;
        }
        __syncthreads();
        #pragma unroll
        for(int u=0;u<2;++u) {
            const int head=tid/32+u*8;
            float localmax=-INFINITY;
            for(int j=lane;j<64;j+=32) {
                if(j>=valid)scores[head*NS+j]=-INFINITY;
                localmax=fmaxf(localmax,scores[head*NS+j]);
            }
            for(int delta=16;delta;delta/=2)localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
            const float nm=fmaxf(maximum[head],localmax),a=expf(maximum[head]-nm);
            float blocksum=0.f;
            for(int j=lane;j<64;j+=32) {
                const float probability=expf(scores[head*NS+j]-nm);
                blocksum+=probability;scores[head*NS+j]=float(T(probability));
            }
            for(int delta=16;delta;delta/=2)blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
            if(lane==0){alpha[head]=a;maximum[head]=nm;denom[head]=denom[head]*a+blocksum;}
        }
        __syncthreads();
        #pragma unroll
        for(int h=0;h<4;++h)for(int d=0;d<2;++d)acc[h*2+d]*=alpha[pv_head+h];
        #pragma unroll 8
        for(int j=0;j<valid;++j) {
            const int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+pv_lane*2;
            const uint32_t packed=*reinterpret_cast<const uint32_t*>(v+off);
            T values[2];__builtin_memcpy(values,&packed,4);
            #pragma unroll
            for(int h=0;h<4;++h) {
                const float probability=scores[(pv_head+h)*NS+j];
                #pragma unroll
                for(int d=0;d<2;++d)acc[h*2+d]=fmaf(probability,float(values[d]),acc[h*2+d]);
            }
        }
        __syncthreads();

    }
    #pragma unroll
    for(int h=0;h<4;++h) {
        const int64_t po=((int64_t(batch)*tokens+query)*hq+kv*16+pv_head+h)*128+pv_lane*2;
        #pragma unroll
        for(int d=0;d<2;++d)out[po+d]=T(acc[h*2+d]/denom[pv_head+h]);
    }
}

extern "C" int personal_nsa_selected_r48_exact16(const void* q,const void* k,const void* v,void* scratch,
 const int32_t* ids,void* out,int b,int t,int hq,int hkv,int width,int bs,float scale,int dtype,void* stream) {
    if(dtype==1&&hq/hkv==16&&bs==64&&t>=1024&&(reinterpret_cast<uintptr_t>(v)&7)==0) {
        nsaSelectedR48Exact16Cuda<<<dim3(b*t,hkv,1),256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,
            (const __nv_bfloat16*)v,ids,(__nv_bfloat16*)out,t,hq,hkv,width,bs,scale);
        return int(cudaGetLastError());
    }
    return personal_nsa_selected_packed_r46_bf16u8(q,k,v,scratch,ids,out,b,t,hq,hkv,width,bs,scale,dtype,stream);
}
