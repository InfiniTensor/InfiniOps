// Personal CUDA prototype for the frozen author's fixed-length prefill selector.
// This does not register or change a public InfiniCore operator ABI.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include "pair_exact.cuh"
#include <mma.h>

template <typename T>
__device__ float read_float(const T *p) { return static_cast<float>(*p); }

template <typename T>
__device__ float scaled_query(const T *p, float scale) {
    // The author multiplies Q by scale, then rounds back to Q dtype BEFORE dot.
    return static_cast<float>(static_cast<T>(read_float(p) * scale));
}

// Literal scalar form of the frozen author's compare/swap network. In a
// descending stage equality swaps too: replacing this by stable argsort or
// a block-index secondary key changes membership at a tied top-k boundary.
__device__ void author_merge(float* x,int* ids,int n,int stage,int order) {
    int tid=threadIdx.x;
    for(int round=0;round<stage;++round) {
        int distance=1<<(stage-round-1);
        float value=0;int id=0;
        if(tid<n) {
            int left=tid&~distance;
            int right=left|distance;
            bool flip=order==2?bool((left>>stage)&1):bool(order);
            bool swap=(x[left]>x[right])!=flip;
            int selected=swap?(tid^distance):tid;
            value=x[selected];id=ids[selected];
        }
        __syncthreads();
        if(tid<n) {x[tid]=value;ids[tid]=id;}
        __syncthreads();
    }
    return;
}

template <typename T>
__global__ __launch_bounds__(128) void nsaPrefillSelectorCuda(
    const T *q, const T *k, const float *lse, int32_t *output,
    int tokens, int qheads, int kvheads, int compressed_blocks,
    int block_size, int topk, float scale) {
    const int token = blockIdx.x % tokens;
    const int batch = blockIdx.x / tokens;
    const int kv = blockIdx.y;
    const int group = qheads / kvheads;
    const int causal_blocks = token / block_size + 1;
    const int tid = threadIdx.x;
    const size_t out_base = (static_cast<size_t>(blockIdx.x) * kvheads + kv) * topk;

    // Each thread owns one (past block, Q head) dot product. Within that
    // dot, dimensions still accumulate as d=0..127 with scalar fmaf.
    // Group sums and the author's sorting network retain their exact order.
    extern __shared__ float importance[];
    float *probabilities=importance+compressed_blocks;
    float *rounded_q=probabilities+compressed_blocks*group;
    const int q_stride=group|1; // odd shared-memory pitch for both G7/G16
    if(causal_blocks>1) {
        for(int i=tid;i<128*group;i+=blockDim.x) {
            const int h=i/128,d=i%128;
            const size_t qb=(size_t(blockIdx.x)*qheads+kv*group+h)*128;
            rounded_q[d*q_stride+h]=scaled_query(q+qb+d,scale);
        }
    }
    __syncthreads();
    for(int task=tid;task<(causal_blocks-1)*group;task+=blockDim.x) {
        const int block=task/group,h=task%group;
        const size_t kb=((size_t(batch)*compressed_blocks+block)*kvheads+kv)*128;
        float dot=0.f;
        if(group==7 && std::is_same<T,__half>::value) {
            // Match the author's FP16 two-product FP32 accumulation boundary.
            // Error-compensated CUDA arithmetic preserves cancellation and
            // midpoint ties before one FP32 rounding per pair.
            for(int d=0;d<128;d+=2) {
                const float p0=__fmul_rn(rounded_q[d*q_stride+h],read_float(k+kb+d));
                const float p1=__fmul_rn(rounded_q[(d+1)*q_stride+h],read_float(k+kb+d+1));
                dot=pair_accumulate_exact(p0,p1,dot);
            }
        } else for(int d=0;d<128;++d)
            dot=fmaf(rounded_q[d*q_stride+h],read_float(k+kb+d),dot);
        const float head_lse=lse[size_t(blockIdx.x)*qheads+kv*group+h];
        const float probability=group==7?exp2f((dot-head_lse)*1.4426950408889634f):expf(dot-head_lse);
        probabilities[block*group+h]=probability<0x1p-126f?0.f:probability;
    }
    __syncthreads();
    for(int block=tid;block<causal_blocks-1;block+=blockDim.x) {
        float score=0.f;
        if(group==7) {
            const float* p=probabilities+block*group;
            float a=__fadd_rn(p[0],p[4]),b=__fadd_rn(p[2],p[6]);
            float c=__fadd_rn(p[1],p[5]),d=p[3];
            score=__fadd_rn(__fadd_rn(a,b),__fadd_rn(c,d));
        } else for(int h=0;h<group;++h)score+=probabilities[block*group+h];
        importance[block]=score;
    }
    if(tid==0)importance[causal_blocks-1]=float(group);
    __syncthreads();
    __shared__ float values[128],previous[128];
    __shared__ int ids[128],previous_ids[128];
    int dims=0;for(int n=block_size;n>1;n>>=1) ++dims;
    for(int start=0;start<causal_blocks;start+=block_size) {
        if(tid<block_size) {
            values[tid]=start+tid<causal_blocks?importance[start+tid]:-1.f;
            ids[tid]=start+tid<causal_blocks?start+tid+1:0;
        }
        __syncthreads();
        for(int stage=1;stage<dims;++stage) author_merge(values,ids,block_size,stage,2);
        if(start!=0) {
            author_merge(values,ids,block_size,dims,0);
            if(tid<block_size/2) {values[tid]=previous[tid];ids[tid]=previous_ids[tid];}
            __syncthreads();
        }
        author_merge(values,ids,block_size,dims,1);
        if(tid<block_size) {previous[tid]=values[tid];previous_ids[tid]=ids[tid];}
        __syncthreads();
    }
    if(tid<topk) output[out_base+tid]=ids[tid]-1;
    return;
}

// Preserve the author's exact compare/swap direction, including tied values.
__device__ __forceinline__ void stage2_merge_wave(float& value,int& id,int n,int stage,int order) {
    const int tid=threadIdx.x;
    for(int round=0;round<stage;++round){
        const int distance=1<<(stage-round-1),left=tid&~distance;
        const float peer=__shfl_xor_sync(0xffffffff,value,distance,n);
        const int peer_id=__shfl_xor_sync(0xffffffff,id,distance,n);
        const bool flip=order==2?bool((left>>stage)&1):bool(order);
        const bool compare=(tid&distance)?peer>value:value>peer;
        const bool swap=compare!=flip;
        if(swap){value=peer;id=peer_id;}
    }
}
template<class T> __global__ __launch_bounds__(256)
void nsaPrefillSelectorG16WaveCuda(const T* q,const T* k,const float* lse,int32_t* output,int tokens,int qheads,int kvheads,int compressed_blocks,int block_size,int topk,float scale) {
    using namespace nvcuda;
    constexpr int DS=136,KS=72,NS=72;
    const int tid=threadIdx.x,warp=tid/warpSize;
    const int token=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int causal_blocks=token/block_size+1,nc=causal_blocks-1;
    const size_t out_base=(size_t(blockIdx.x)*kvheads+kv)*topk;
    extern __shared__ float importance[];
    float* probabilities=importance+compressed_blocks;
    __shared__ __align__(32) T qs[16*DS];
    __shared__ __align__(32) float scores[16*NS];
    for(int i=tid;i<16*128;i+=256) {
        const int h=i/128,d=i%128;const int gd=d;
        const size_t qb=(size_t(blockIdx.x)*qheads+kv*16+h)*128+gd;
        qs[h*DS+d]=T(scaled_query(q+qb,scale));
    }
    __syncthreads();
    unsigned long long qkmap=0;
    T* marker=reinterpret_cast<T*>(scores);
    for(int i=tid;i<256;i+=256)marker[(i/16)*NS+i%16]=T(float(i));
    __syncthreads();
    if(warp<4){
        wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> map;
        wmma::load_matrix_sync(map,marker,NS);
        static_assert(map.num_elements<=8,"mapping fits64 bits");
        for(int i=0;i<map.num_elements;++i)qkmap|=static_cast<unsigned long long>(float(map.x[i]))<<(i*8);
    }
    __syncthreads();
    for(int start=0;start<nc;start+=64) {
        wmma::fragment<wmma::accumulator,16,16,16,float> c;
        wmma::fill_fragment(c,0.f);
        if(warp<4){
            #pragma unroll
            for(int di=0;di<8;++di){
                wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> b;
                wmma::load_matrix_sync(a,qs+di*16,DS);
                #pragma unroll
                for(int i=0;i<b.num_elements;++i){
                    const int m=(qkmap>>(i*8))&255,block=start+warp*16+m/16,d=di*16+m%16;
                    b.x[i]=block<nc?k[((size_t(batch)*compressed_blocks+block)*kvheads+kv)*128+d]:T(0.f);
                }
                wmma::mma_sync(c,a,b,c);
            }
        }
        if(warp<4)wmma::store_matrix_sync(scores+warp*16,c,NS,wmma::mem_row_major);
        __syncthreads();
        for(int task=tid;task<64*16;task+=256) {
            const int h=task%16,j=task/16,block=start+j;
            if(block<nc) {
                const float head_lse=lse[size_t(blockIdx.x)*qheads+kv*16+h];
                const float probability=exp2f((scores[h*NS+j]-head_lse)*1.4426950408889634f);
                float p=probability<0x1p-126f?0.f:probability;
                #pragma unroll
                for(int delta=8;delta;delta/=2)p=__fadd_rn(p,__shfl_xor_sync(0xffffffff,p,delta,16));
                if(h==0)importance[block]=p;
            }
        }
        __syncthreads();
    }
    if(tid==0)importance[nc]=16.f;
    __syncthreads();
    {
        if(tid<block_size) {
            float previous=-1.f;int previous_id=0;
            int dims=0;for(int n=block_size;n>1;n>>=1)++dims;
            for(int start=0;start<causal_blocks;start+=block_size) {
                float value=start+tid<causal_blocks?importance[start+tid]:-1.f;
                int id=start+tid<causal_blocks?start+tid+1:0;
                for(int stage=1;stage<dims;++stage)stage2_merge_wave(value,id,block_size,stage,2);
                if(start!=0) {
                    stage2_merge_wave(value,id,block_size,dims,0);
                    if(tid<block_size/2){value=previous;id=previous_id;}
                }
                stage2_merge_wave(value,id,block_size,dims,1);
                previous=value;previous_id=id;
            }
            if(tid<topk)output[out_base+tid]=previous_id-1;
        }
        return;
    }
}
template<class T> __global__ __launch_bounds__(256)
void nsaPrefillSelectorG16WmmaCuda(const T* q,const T* k,const float* lse,int32_t* output,int tokens,int qheads,int kvheads,int compressed_blocks,int block_size,int topk,float scale) {
    using namespace nvcuda;
    constexpr int DS=136,KS=72,NS=72;
    const int tid=threadIdx.x,warp=tid/warpSize;
    const int token=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int causal_blocks=token/block_size+1,nc=causal_blocks-1;
    const size_t out_base=(size_t(blockIdx.x)*kvheads+kv)*topk;
    extern __shared__ float importance[];
    float* probabilities=importance+compressed_blocks;
    __shared__ __align__(32) T qs[16*DS],ks[64*KS];
    __shared__ __align__(32) float scores[16*NS];
    for(int i=tid;i<16*128;i+=256) {
        const int h=i/128,d=i%128;const int gd=d;
        const size_t qb=(size_t(blockIdx.x)*qheads+kv*16+h)*128+gd;
        qs[h*DS+d]=T(scaled_query(q+qb,scale));
    }
    __syncthreads();
    for(int start=0;start<nc;start+=64) {
        wmma::fragment<wmma::accumulator,16,16,16,float> c;
        wmma::fill_fragment(c,0.f);
        for(int half=0;half<2;++half) {
            for(int i=tid;i<64*64;i+=256) {
                const int block=start+i/64,d=half*64+i%64;const int gd=d;
                const size_t kb=((size_t(batch)*compressed_blocks+block)*kvheads+kv)*128+gd;
                ks[(i/64)*KS+i%64]=block<nc?k[kb]:T(0.f);
            }
            __syncthreads();
            if(warp<4)for(int di=0;di<4;++di) {
                wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> b;
                wmma::load_matrix_sync(a,qs+half*64+di*16,DS);
                wmma::load_matrix_sync(b,ks+warp*16*KS+di*16,KS);
                wmma::mma_sync(c,a,b,c);
            }
            __syncthreads();
        }
        if(warp<4)wmma::store_matrix_sync(scores+warp*16,c,NS,wmma::mem_row_major);
        __syncthreads();
        for(int task=tid;task<64*16;task+=256) {
            const int h=task%16,j=task/16,block=start+j;
            if(block<nc) {
                const float head_lse=lse[size_t(blockIdx.x)*qheads+kv*16+h];
                const float probability=exp2f((scores[h*NS+j]-head_lse)*1.4426950408889634f);
                probabilities[block*16+h]=probability<0x1p-126f?0.f:probability;
            }
        }
        __syncthreads();
    }
    for(int block=tid;block<nc;block+=256) {
        float p[16];
        #pragma unroll
        for(int h=0;h<16;++h)p[h]=probabilities[block*16+h];
        #pragma unroll
        for(int delta=8;delta;delta/=2) {
            #pragma unroll
            for(int h=0;h<delta;++h)p[h]=__fadd_rn(p[h],p[h+delta]);
        }
        importance[block]=p[0];
    }
    if(tid==0)importance[nc]=16.f;
    __syncthreads();
    __shared__ float values[128],previous[128];
    __shared__ int ids[128],previous_ids[128];
    int dims=0;for(int n=block_size;n>1;n>>=1) ++dims;
    for(int start=0;start<causal_blocks;start+=block_size) {
        if(tid<block_size) {
            values[tid]=start+tid<causal_blocks?importance[start+tid]:-1.f;
            ids[tid]=start+tid<causal_blocks?start+tid+1:0;
        }
        __syncthreads();
        for(int stage=1;stage<dims;++stage) author_merge(values,ids,block_size,stage,2);
        if(start!=0) {
            author_merge(values,ids,block_size,dims,0);
            if(tid<block_size/2) {values[tid]=previous[tid];ids[tid]=previous_ids[tid];}
            __syncthreads();
        }
        author_merge(values,ids,block_size,dims,1);
        if(tid<block_size) {previous[tid]=values[tid];previous_ids[tid]=ids[tid];}
        __syncthreads();
    }
    if(tid<topk) output[out_base+tid]=ids[tid]-1;
    return;
}
extern "C" int personal_nsa_prefill_selector(
    const void *q, const void *k, const float *lse, int32_t *output,
    int batch, int tokens, int qheads, int kvheads, int dim, int compressed_blocks,
    int block_size, int topk, float scale, int dtype, void *stream) {
    if (!q || !k || !lse || !output || batch <= 0 || tokens <= 0 || tokens > 8192 ||
        qheads <= 0 || kvheads <= 0 || qheads % kvheads || dim != 128 ||
        block_size < 2 || block_size > 128 || (block_size & (block_size-1)) || topk > block_size/2 || topk <= 0 || topk > 32 || !std::isfinite(scale) ||
        compressed_blocks != (tokens + block_size - 1) / block_size ||
        qheads / kvheads > 32 || (dtype != 0 && dtype != 1)) return -1;
    const int group = qheads / kvheads;
    if(group==16 && compressed_blocks<=512) {
        const size_t shared=sizeof(float)*compressed_blocks*17;
        dim3 grid(batch*tokens,kvheads);
        if(block_size==64){
            int device=0,wave=0;auto err=cudaGetDevice(&device);if(err!=cudaSuccess)return int(err);err=cudaDeviceGetAttribute(&wave,cudaDevAttrWarpSize,device);if(err!=cudaSuccess)return int(err);
            if(wave==64){
                if(dtype==0)nsaPrefillSelectorG16WaveCuda<<<grid,256,sizeof(float)*compressed_blocks,static_cast<cudaStream_t>(stream)>>>((const __half*)q,(const __half*)k,lse,output,tokens,qheads,kvheads,compressed_blocks,block_size,topk,scale);
                else nsaPrefillSelectorG16WaveCuda<<<grid,256,sizeof(float)*compressed_blocks,static_cast<cudaStream_t>(stream)>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,lse,output,tokens,qheads,kvheads,compressed_blocks,block_size,topk,scale);
                return int(cudaGetLastError());
            }
        }
        if(dtype==0)nsaPrefillSelectorG16WmmaCuda<<<grid,256,shared,static_cast<cudaStream_t>(stream)>>>((const __half*)q,(const __half*)k,lse,output,tokens,qheads,kvheads,compressed_blocks,block_size,topk,scale);
        else nsaPrefillSelectorG16WmmaCuda<<<grid,256,shared,static_cast<cudaStream_t>(stream)>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,lse,output,tokens,qheads,kvheads,compressed_blocks,block_size,topk,scale);
        return int(cudaGetLastError());
    }

    const int threads = 128;
    const dim3 grid(batch * tokens, kvheads);
    const size_t shared_bytes = sizeof(float)*(compressed_blocks*(group+1)+128*(group|1));
    if (dtype == 0) {
        nsaPrefillSelectorCuda<__half><<<grid, threads, shared_bytes, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const __half *>(q), static_cast<const __half *>(k), lse, output,
            tokens, qheads, kvheads, compressed_blocks, block_size, topk, scale);
    } else {
        nsaPrefillSelectorCuda<__nv_bfloat16><<<grid, threads, shared_bytes, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const __nv_bfloat16 *>(q), static_cast<const __nv_bfloat16 *>(k), lse, output,
            tokens, qheads, kvheads, compressed_blocks, block_size, topk, scale);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" int personal_nsa_resources_selector_wave(int* v){cudaFuncAttributes a;auto e=cudaFuncGetAttributes(&a,nsaPrefillSelectorG16WaveCuda<__nv_bfloat16>);if(e!=cudaSuccess)return int(e);int blocks=0;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,nsaPrefillSelectorG16WaveCuda<__nv_bfloat16>,256,37*sizeof(float));v[0]=a.numRegs;v[1]=a.sharedSizeBytes;v[2]=a.localSizeBytes;v[3]=blocks;return int(e);}
