
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>
extern "C" int personal_nsa_selected_r48_exact16(const void*,const void*,const void*,void*,const int32_t*,void*,int,int,int,int,int,int,float,int,void*);


static __device__ __forceinline__ float r50_selected_exp2_final(float x){return exp2f(__fmul_rn(x,1.4426950408889634f));}
template<class T> __global__ __launch_bounds__(256)
void r50_fragment_pack_matrix_clean2(const T* k,const T* v,T* pk,T* pv,int tokens,int hkv,int batches){
    using namespace nvcuda;const int tid=threadIdx.x,warp=tid/warpSize,lane=tid%warpSize,blocks=(tokens+15)/16;
    __shared__ __align__(32) T marker[16*24];
    for(int i=tid;i<256;i+=256)marker[(i/16)*24+i%16]=T(float(i));
    __syncthreads();
    const int task=blockIdx.x*4+warp,total=batches*hkv*blocks*8;if(task>=total)return;
    const int di=task%8,block=(task/8)%blocks,kv=(task/(8*blocks))%hkv,batch=task/(8*blocks*hkv);
    const int64_t target=int64_t(task)*256+lane*4;
    if(1){
        wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> map,b;
        static_assert(map.num_elements==4,"native pack contract");wmma::load_matrix_sync(map,marker,24);
        #pragma unroll
        for(int i=0;i<map.num_elements;++i){const int m=int(float(map.x[i])),pos=block*16+m/16,d=di*16+m%16;b.x[i]=pos<tokens?k[((int64_t(batch)*tokens+pos)*hkv+kv)*128+d]:T(0.f);}
        uint64_t words;__builtin_memcpy(&words,b.x,sizeof(b.x));*reinterpret_cast<uint64_t*>(pk+target)=words;
    }
    if(1){
        wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> map,b;
        static_assert(map.num_elements==4,"native pack contract");wmma::load_matrix_sync(map,marker,24);
        #pragma unroll
        for(int i=0;i<map.num_elements;++i){const int m=int(float(map.x[i])),pos=block*16+m/16,d=di*16+m%16;b.x[i]=pos<tokens?v[((int64_t(batch)*tokens+pos)*hkv+kv)*128+d]:T(0.f);}
        uint64_t words;__builtin_memcpy(&words,b.x,sizeof(b.x));*reinterpret_cast<uint64_t*>(pv+target)=words;
    }
}
template<class T> __global__ __launch_bounds__(256)
void nsaSelectedR50matrix_clean2Cuda(const T* q,const T* k,const T* v,const int32_t* ids,T* out,const T* packedk,const T* packedv,
 int tokens,int hq,int hkv,int width,int bs,float scale) {
    using namespace nvcuda;
    constexpr int DS=136,NS=72;
    const int tid=threadIdx.x,warp=tid/warpSize,lane=tid%32;
    const int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ __align__(32) T qs[16*DS],ps[16*NS];
    __shared__ __align__(32) float scores[16*NS];
    __shared__ float maximum[16],denom[16],alpha[16];
    for(int i=tid;i<16*128;i+=256) {
        const int head=i/128,d=i%128;
        qs[head*DS+d]=T(float(q[((int64_t(batch)*tokens+query)*hq+kv*16+head)*128+d])*scale);
    }
    if(tid<16){maximum[tid]=-INFINITY;denom[tid]=0.f;}
    __syncthreads();
    unsigned long long qkmap=0,pvmap=0;unsigned int rowmap=0;
    T* marker=reinterpret_cast<T*>(scores);
    for(int i=tid;i<256;i+=256)marker[(i/16)*NS+i%16]=T(float(i));
    __syncthreads();
    if(warp<4){
        wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> kb;
        wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> vb;
        static_assert(kb.num_elements<=8&&vb.num_elements<=8,"maps fit64 bits");
        wmma::load_matrix_sync(kb,marker,NS);wmma::load_matrix_sync(vb,marker,NS);
        for(int i=0;i<kb.num_elements;++i)qkmap|=static_cast<unsigned long long>(float(kb.x[i]))<<(i*8);
        for(int i=0;i<vb.num_elements;++i)pvmap|=static_cast<unsigned long long>(float(vb.x[i]))<<(i*8);
    }
    __syncthreads();
    for(int i=tid;i<256;i+=256)scores[(i/16)*NS+i%16]=float(i/16);
    __syncthreads();
    if(warp<4){
        wmma::fragment<wmma::accumulator,16,16,16,float> rows;
        wmma::load_matrix_sync(rows,scores,NS,wmma::mem_row_major);
        static_assert(rows.num_elements<=8,"row map fits32 bits");
        for(int i=0;i<rows.num_elements;++i)rowmap|=unsigned(rows.x[i])<<(i*4);
    }
    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> qcache[8];
    if(warp<4){
        #pragma unroll
        for(int i=0;i<8;++i)wmma::load_matrix_sync(qcache[i],qs+i*16,DS);
    }
    wmma::fragment<wmma::accumulator,16,16,16,float> output[2];
    if(warp<4)for(int h=0;h<2;++h)wmma::fill_fragment(output[h],0.f);
    __syncthreads();
    for(int slot=0;slot<width;++slot) {
        const int id=ids[io+slot];
        if(id<0||int64_t(id)*64>query)continue;
        const int start=id*64,valid=min(64,query+1-start);
        if(warp<4){
            wmma::fragment<wmma::accumulator,16,16,16,float> c;wmma::fill_fragment(c,0.f);
            #pragma unroll
            for(int di=0;di<8;++di){
                wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> b;
                const int block=(start/16)+warp,blocks=(tokens+15)/16;
                const int64_t off=(((int64_t(batch)*hkv+kv)*blocks+block)*8+di)*256+(tid%warpSize)*b.num_elements;
                const uint64_t words=*reinterpret_cast<const uint64_t*>(packedk+off);__builtin_memcpy(b.x,&words,sizeof(b.x));
                wmma::mma_sync(c,qcache[di],b,c);
            }
            wmma::store_matrix_sync(scores+warp*16,c,NS,wmma::mem_row_major);
        }
        __syncthreads();
        #pragma unroll
        for(int u=0;u<2;++u){
            const int head=tid/32+u*8;float localmax=-INFINITY;
            for(int j=lane;j<64;j+=32){if(j>=valid)scores[head*NS+j]=-INFINITY;localmax=fmaxf(localmax,scores[head*NS+j]);}
            for(int delta=16;delta;delta/=2)localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
            const float nm=fmaxf(maximum[head],localmax),a=r50_selected_exp2_final(maximum[head]-nm);float blocksum=0.f;
            for(int j=lane;j<64;j+=32){const float p=r50_selected_exp2_final(scores[head*NS+j]-nm);blocksum+=p;ps[head*NS+j]=T(p);}
            for(int delta=16;delta;delta/=2)blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
            if(lane==0){alpha[head]=a;maximum[head]=nm;denom[head]=denom[head]*a+blocksum;}
        }
        __syncthreads();
        if(warp<4){
            for(int h=0;h<2;++h)for(int i=0;i<output[h].num_elements;++i)output[h].x[i]*=alpha[(rowmap>>(i*4))&15];
            #pragma unroll
            for(int half=0;half<2;++half){
                #pragma unroll
                for(int j=0;j<64;j+=16){
                    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> b;
                    wmma::load_matrix_sync(a,ps+j,NS);
                    const int block=(start+j)/16,blocks=(tokens+15)/16,di=half*4+warp;
                    const int64_t off=(((int64_t(batch)*hkv+kv)*blocks+block)*8+di)*256+(tid%warpSize)*b.num_elements;
                    const uint64_t words=*reinterpret_cast<const uint64_t*>(packedv+off);__builtin_memcpy(b.x,&words,sizeof(b.x));
                    if(j+16>valid){
                        #pragma unroll
                        for(int i=0;i<b.num_elements;++i)if(j+int((pvmap>>(i*8))&255)/16>=valid)b.x[i]=T(0.f);
                    }
                    wmma::mma_sync(output[half],a,b,output[half]);
                }
            }
        }
        __syncthreads();

    }
    for(int h=0;h<2;++h) {
        if(warp<4)wmma::store_matrix_sync(scores+warp*16,output[h],NS,wmma::mem_row_major);
        __syncthreads();
        for(int i=tid;i<16*64;i+=256) {
            const int head=i/64,d=h*64+i%64;
            out[((int64_t(batch)*tokens+query)*hq+kv*16+head)*128+d]=T(scores[head*NS+i%64]/denom[head]);
        }
        __syncthreads();
    }
}
extern "C" size_t personal_nsa_selected_scratch_r50_matrix_clean2(int b,int t,int hq,int hkv,int width,int bs,int dtype){
    if(b<1||b>8||t<1024||t>8192||hq<1||hq>1024||hkv<1||hkv>32||hq%hkv||hq/hkv!=16||width<1||width>32||bs!=64||dtype!=1)return 0;
    return size_t(b)*hkv*((t+15)/16)*128*16*sizeof(__nv_bfloat16)*2;
}
extern "C" int personal_nsa_selected_r50_matrix_clean2(const void* q,const void* k,const void* v,void* scratch,const int32_t* ids,void* out,int b,int t,int hq,int hkv,int width,int bs,float scale,int dtype,void* stream){
    size_t bytes=personal_nsa_selected_scratch_r50_matrix_clean2(b,t,hq,hkv,width,bs,dtype);
    if(!bytes||!scratch)return personal_nsa_selected_r48_exact16(q,k,v,scratch,ids,out,b,t,hq,hkv,width,bs,scale,dtype,stream);
    auto pk=(__nv_bfloat16*)scratch;auto pv=pk+bytes/4;
    const int tasks=b*hkv*((t+15)/16)*8;
    r50_fragment_pack_matrix_clean2<<<(tasks+3)/4,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)k,(const __nv_bfloat16*)v,pk,pv,t,hkv,b);
    auto e=cudaGetLastError();if(e!=cudaSuccess)return int(e);
    nsaSelectedR50matrix_clean2Cuda<<<dim3(b*t,hkv,1),256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,ids,(__nv_bfloat16*)out,pk,pv,t,hq,hkv,width,bs,scale);
    return int(cudaGetLastError());
}

extern "C" int personal_nsa_resources_r50_matrix_clean2(int* v){cudaFuncAttributes a;auto e=cudaFuncGetAttributes(&a,nsaSelectedR50matrix_clean2Cuda<__nv_bfloat16>);if(e!=cudaSuccess)return int(e);int blocks=0;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,nsaSelectedR50matrix_clean2Cuda<__nv_bfloat16>,256,0);v[0]=a.numRegs;v[1]=a.sharedSizeBytes;v[2]=a.localSizeBytes;v[3]=blocks;return int(e);}
