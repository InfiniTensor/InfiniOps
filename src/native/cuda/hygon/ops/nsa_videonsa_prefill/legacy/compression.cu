#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#include <type_traits>
#include <mma.h>
static __device__ float approxWindowP(float score, float maximum, float scale, float* pv) {
 float c=__uint_as_float(0x3e6e28f6u);
 float z=fminf(1.f,fmaxf(0.f,fmaf(score,scale*(1.f/64.f),fmaf(-maximum,scale*(1.f/64.f),c))));
 unsigned short b=(unsigned short)floorf(z*65535.f+0.5f);
 half a=__ushort_as_half(b),d=__ushort_as_half((unsigned short)(b+512));
 half r=__float2half_rn(fmaf(__half2float(a),__half2float(__ushort_as_half(0x3da8)),__half2float(d)));
 unsigned short bits=__half_as_ushort(r);
 *pv=__bfloat162float(__ushort_as_bfloat16((unsigned short)(((bits>>3)<0x80)?0:((bits>>3)+0x3800))));
 return __half2float(r);
}
static __device__ float halfRound(float x){volatile unsigned short b=__half_as_ushort(__float2half_rn(x));return __half2float(__ushort_as_half(b));}
__device__ __forceinline__ unsigned int r49_keep_bits(unsigned int bits) {
#if defined(__AMDGCN__)
    asm volatile("" : "+v"(bits));
#else
    asm volatile("" : "+r"(bits));
#endif
    return bits;
}
__device__ __forceinline__ float r49_half_round(float x) {
    const unsigned int bits=r49_keep_bits(__half_as_ushort(__float2half_rn(x)));
    return __half2float(__ushort_as_half((unsigned short)bits));
}
template<class G> __device__ __forceinline__ float r49_gate_round(float x) {
    if constexpr(std::is_same<G,half>::value)return r49_half_round(x);
    else if constexpr(std::is_same<G,__nv_bfloat16>::value) {
        const unsigned int bits=r49_keep_bits(__bfloat16_as_ushort(__float2bfloat16_rn(x)));
        return __bfloat162float(__ushort_as_bfloat16((unsigned short)bits));
    } else {
        return __uint_as_float(r49_keep_bits(__float_as_uint(x)));
    }
}
template<class T,class G> __global__ void nsaGateR49Reg(const T* oc,const T* os,const T* ow,
 const G* gc,const G* gs,const G* gw,G* out,int64_t elements) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<elements) {
        const int64_t row=i/128;
        const float pc=r49_gate_round<G>(__fmul_rn(float(oc[i]),float(gc[row])));
        const float ps=r49_gate_round<G>(__fmul_rn(float(os[i]),float(gs[row])));
        const float pw=r49_gate_round<G>(__fmul_rn(float(ow[i]),float(gw[row])));
        const float partial=r49_gate_round<G>(__fadd_rn(pc,ps));
        out[i]=G(__fadd_rn(partial,pw));
    }
}
extern "C" int personal_nsa_gate_r49_reg(const void* oc,const void* os,const void* ow,
 const void* gc,const void* gs,const void* gw,void* out,int64_t n,int dtype,int gate32,void* stream) {
    const dim3 grid((n+255)/256);
    if(dtype==0&&gate32==0)nsaGateR49Reg<<<grid,256,0,(cudaStream_t)stream>>>((const half*)oc,(const half*)os,(const half*)ow,(const half*)gc,(const half*)gs,(const half*)gw,(half*)out,n);
    else if(dtype==1&&gate32==0)nsaGateR49Reg<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)oc,(const __nv_bfloat16*)os,(const __nv_bfloat16*)ow,(const __nv_bfloat16*)gc,(const __nv_bfloat16*)gs,(const __nv_bfloat16*)gw,(__nv_bfloat16*)out,n);
    else if(dtype==0)nsaGateR49Reg<<<grid,256,0,(cudaStream_t)stream>>>((const half*)oc,(const half*)os,(const half*)ow,(const float*)gc,(const float*)gs,(const float*)gw,(float*)out,n);
    else nsaGateR49Reg<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)oc,(const __nv_bfloat16*)os,(const __nv_bfloat16*)ow,(const float*)gc,(const float*)gs,(const float*)gw,(float*)out,n);
    return int(cudaGetLastError());
}
extern "C" int personal_nsa_branch_r47_wsoft(const void*,const void*,const void*,void*,float*,int,int,int,int,int,int,int,int,int,void*);
extern "C" int personal_nsa_branch_r47_creg(const void*,const void*,const void*,void*,float*,int,int,int,int,int,int,int,int,int,void*);
template<class T> __global__ __launch_bounds__(256)
void nsaAttentionR49wdkv32Cuda(const T* q,const T* k,const T* v,T* out,
                             int tokens,int hq,int hkv,int window) {
    using namespace nvcuda;
    constexpr int DS=136,NS=72,KS=72; // Padding avoids power-of-two shared-memory strides.
    int tid=threadIdx.x,warp=tid/warpSize,query0=blockIdx.x*32,batch=blockIdx.y,head=blockIdx.z;
    int kv=head/(hq/hkv);
    __shared__ __align__(32) union {T query[32*DS];float score[32*NS];} query_score;
    T* qs=query_score.query;float* scores=query_score.score;
    __shared__ __align__(32) T ps[32*NS];
     // K is no longer needed when P*V begins.

    __shared__ float maximum[32],denom[32],rescale[32];
    for(int i=tid;i<32*128;i+=256) {
        int row=query0+i/128,d=i%128;int gd=d;gd=(d/32)*32+((d%16)/4)*8+d%4+((d%32)/16)*4;
        qs[(i/128)*DS+d]=row<tokens?q[((int64_t(batch)*tokens+row)*hq+head)*128+gd]:T(0.f);

    }
    if(tid<32) {maximum[tid]=-INFINITY;denom[tid]=0;}
    __syncthreads();
    wmma::fragment<wmma::accumulator,16,16,16,float> output[4];
    if(warp<4)for(int h=0;h<4;++h)wmma::fill_fragment(output[h],0.f);
    int earliest=window>0?max(0,(query0/128)*128-window+1):0;
    int latest=min(tokens-1,(query0/128)*128+127);
    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> qcache[2][8];
    if(warp<4) {
        #pragma unroll
        for(int sub=0;sub<2;++sub) {
            #pragma unroll
            for(int di=0;di<8;++di)wmma::load_matrix_sync(qcache[sub][di],qs+sub*16*DS+di*16,DS);
        }
    }
    __syncthreads();
    unsigned long long qk_bmap=0,pv_bmap=0;
    T* operand_marker=reinterpret_cast<T*>(scores);
    for(int i=tid;i<256;i+=256)operand_marker[(i/16)*NS+i%16]=T(float(i));
    __syncthreads();
    if(warp<4) {
        wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> qk;
        wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> pv;
        static_assert(qk.num_elements<=8&&pv.num_elements<=8,"operand maps fit64 bits");
        wmma::load_matrix_sync(qk,operand_marker,NS);wmma::load_matrix_sync(pv,operand_marker,NS);
        #pragma unroll
        for(int i=0;i<qk.num_elements;++i)qk_bmap|=static_cast<unsigned long long>(float(qk.x[i]))<<(i*8);
        #pragma unroll
        for(int i=0;i<pv.num_elements;++i)pv_bmap|=static_cast<unsigned long long>(float(pv.x[i]))<<(i*8);
    }
    __syncthreads();
    constexpr float scale_log2=float(double(0.08838834764831845f)*1.4426950408889634);
    for(int start=latest/64*64;start>=earliest/64*64;start-=64) {
        const int query_end=min(tokens-1,query0+31);
        const int query_begin=window>0?max(0,query0-window+1):0;
        const bool empty=start>query_end || start+63<query_begin;
        if(!empty) {
            if(warp<4) {
                wmma::fragment<wmma::accumulator,16,16,16,float> dots[2];
                #pragma unroll
                for(int sub=0;sub<2;++sub)wmma::fill_fragment(dots[sub],0.f);
                #pragma unroll
                for(int di=0;di<8;++di) {
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> b;
                    #pragma unroll
                    for(int i=0;i<b.num_elements;++i) {
                        const int m=(qk_bmap>>(i*8))&255,pos=start+warp*16+m/16,d=di*16+m%16;
                        const int gd=(d/32)*32+((d%16)/4)*8+d%4+((d%32)/16)*4;
                        b.x[i]=pos<tokens?k[((int64_t(batch)*tokens+pos)*hkv+kv)*128+gd]:T(0.f);
                    }
                    #pragma unroll
                    for(int sub=0;sub<2;++sub)wmma::mma_sync(dots[sub],qcache[sub][di],b,dots[sub]);
                }
                #pragma unroll
                for(int sub=0;sub<2;++sub)wmma::store_matrix_sync(scores+sub*16*NS+warp*16,dots[sub],NS,wmma::mem_row_major);
            }
        } else {
            for(int i=tid;i<32*64;i+=256)scores[(i/64)*NS+i%64]=-INFINITY;
        }
        __syncthreads();
        for(int sub=0;sub<2;++sub)
        {
            const int row=tid/16+sub*16,lane=tid%16,query=query0+row;
            const int begin=window>0?max(0,query-window+1):0;
            float mx=-INFINITY;
            for(int j=lane;j<64;j+=16) {
                const int pos=start+j;
                if(query>=tokens||pos<begin||pos>query)scores[row*NS+j]=-INFINITY;
                mx=fmaxf(mx,scores[row*NS+j]);
            }
            for(int delta=8;delta;delta/=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,delta,16));
            const float nm=fmaxf(maximum[row],mx);
            const float alpha=isfinite(nm)?exp2f((maximum[row]-nm)*scale_log2):1.f;
            for(int j=lane;j<64;j+=16) {
                float prob=0.f,pv=0.f;
                if(isfinite(nm)){prob=approxWindowP(scores[row*NS+j],nm,scale_log2,&pv);if(std::is_same<T,half>::value)pv=prob;}
                scores[row*NS+j]=prob;ps[row*NS+j]=T(pv);
            }
            if(lane==0){maximum[row]=nm;rescale[row]=alpha;}
        }
        __syncthreads();
        if(tid<128) {
            const int row=tid/4,lane=tid%4;
            float lo=0.f,hi=0.f;
            for(int tile=0;tile<4;++tile)for(int pair=0;pair<2;++pair){const int j=tile*16+lane*4+pair*2;lo=r49_half_round(lo+scores[row*NS+j]);hi=r49_half_round(hi+scores[row*NS+j+1]);}
            float sum=r49_half_round(lo+hi);
            for(int delta=2;delta;delta/=2)sum+=__shfl_xor_sync(0xffffffff,sum,delta,4);
            if(lane==0)denom[row]=denom[row]*rescale[row]+sum;
        }
        __syncthreads();
        for(int i=tid;i<32*16;i+=256)scores[(i/16)*NS+i%16]=rescale[i/16];
        __syncthreads();
        if(warp<4)for(int sub=0;sub<2;++sub) {
            wmma::fragment<wmma::accumulator,16,16,16,float> factor;
            wmma::load_matrix_sync(factor,scores+sub*16*NS,NS,wmma::mem_row_major);
            for(int h=0;h<2;++h)for(int i=0;i<factor.num_elements;++i)output[sub*2+h].x[i]*=factor.x[i];
        }
        __syncthreads();
        if(!empty&&warp<4) {
            #pragma unroll
            for(int half=0;half<2;++half) {
                #pragma unroll
                for(int j=0;j<64;j+=16) {
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> b;
                    #pragma unroll
                    for(int i=0;i<b.num_elements;++i) {
                        const int m=(pv_bmap>>(i*8))&255,pos=start+j+m/16,d=half*64+warp*16+m%16;
                        b.x[i]=pos<tokens?v[((int64_t(batch)*tokens+pos)*hkv+kv)*128+d]:T(0.f);
                    }
                    #pragma unroll
                    for(int sub=0;sub<2;++sub) {
                        wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                        wmma::load_matrix_sync(a,ps+sub*16*NS+j,NS);
                        wmma::mma_sync(output[sub*2+half],a,b,output[sub*2+half]);
                    }
                }
            }
        }
        __syncthreads();
    }
    for(int h=0;h<2;++h) {
        if(warp<4) for(int sub=0;sub<2;++sub)wmma::store_matrix_sync(scores+sub*16*NS+warp*16,output[sub*2+h],NS,wmma::mem_row_major);
        __syncthreads();
        for(int i=tid;i<32*64;i+=256) {
            int row=query0+i/64,d=h*64+i%64;
            if(row<tokens)out[((int64_t(batch)*tokens+row)*hq+head)*128+d]=T(__bfloat162float(__ushort_as_bfloat16((unsigned short)((__float_as_uint(scores[(i/64)*NS+i%64]/denom[i/64]*1.00390625f)+0x8000u)>>16))));
        }
        __syncthreads();
    }
    return;
}
extern "C" int personal_nsa_branch_r49_wdkv32(const void* q,const void* k,const void* v,void* out,float* lse,
 int b,int t,int n,int hq,int hkv,int bs,int window,int kind,int dtype,void* stream) {
    if(kind==1&&dtype==1&&t>=1024&&hq/hkv==16) {
        int device=0,sm=0;auto e=cudaGetDevice(&device);if(e!=cudaSuccess)return int(e);
        e=cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device);if(e!=cudaSuccess)return int(e);
        if(int64_t((t+63)/64)*b*hq<int64_t(sm)*4)return personal_nsa_branch_r47_wsoft(q,k,v,out,lse,b,t,n,hq,hkv,bs,window,kind,dtype,stream);
        nsaAttentionR49wdkv32Cuda<<<dim3((t+32-1)/32,b,hq),256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,t,hq,hkv,window);
        return int(cudaGetLastError());
    }
    return personal_nsa_branch_r47_wsoft(q,k,v,out,lse,b,t,n,hq,hkv,bs,window,kind,dtype,stream);
}
extern "C" int personal_nsa_resources_r49_wdkv32(int* values) {
    cudaFuncAttributes a;auto e=cudaFuncGetAttributes(&a,nsaAttentionR49wdkv32Cuda<__nv_bfloat16>);
    if(e!=cudaSuccess)return int(e);int blocks=0;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,nsaAttentionR49wdkv32Cuda<__nv_bfloat16>,256,0);
    values[0]=a.numRegs;values[1]=a.sharedSizeBytes;values[2]=a.localSizeBytes;values[3]=blocks;return int(e);
}
template<class T> __global__ __launch_bounds__(256)
void nsaAttentionR49cdkv32Cuda(const T* q,const T* k,const T* v,T* out,
                             int tokens,int keytokens,int hq,int hkv,int bs,float* lse) {
    using namespace nvcuda;
    constexpr int DS=136,NS=72; // Padding avoids power-of-two shared-memory strides.
    int tid=threadIdx.x,warp=tid/warpSize,query0=blockIdx.x*32,batch=blockIdx.y,head=blockIdx.z;
    int kv=head/(hq/hkv);
    __shared__ __align__(32) union {T query[32*DS];float score[32*NS];} query_score;
    T* qs=query_score.query;float* scores=query_score.score;
    __shared__ __align__(32) T ps[32*NS];
     // K is no longer needed when P*V begins.

    __shared__ float maximum[32],denom[32],rescale[32];
    for(int i=tid;i<32*128;i+=256) {
        int row=query0+i/128,d=i%128;
        qs[(i/128)*DS+d]=row<tokens?q[((int64_t(batch)*tokens+row)*hq+head)*128+d]:T(0.f);
    }
    if(tid<32) {maximum[tid]=-INFINITY;denom[tid]=0;}
    __syncthreads();
    wmma::fragment<wmma::accumulator,16,16,16,float> output[4];
    if(warp<4)for(int h=0;h<4;++h)wmma::fill_fragment(output[h],0.f);
    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> qcache[2][8];
    if(warp<4) {
        #pragma unroll
        for(int sub=0;sub<2;++sub) {
            #pragma unroll
            for(int di=0;di<8;++di)wmma::load_matrix_sync(qcache[sub][di],qs+sub*16*DS+di*16,DS);
        }
    }
    __syncthreads();
    unsigned long long qk_bmap=0,pv_bmap=0;
    T* operand_marker=reinterpret_cast<T*>(scores);
    for(int i=tid;i<256;i+=256)operand_marker[(i/16)*NS+i%16]=T(float(i));
    __syncthreads();
    if(warp<4) {
        wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> qk;
        wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> pv;
        static_assert(qk.num_elements<=8&&pv.num_elements<=8,"operand maps fit64 bits");
        wmma::load_matrix_sync(qk,operand_marker,NS);wmma::load_matrix_sync(pv,operand_marker,NS);
        #pragma unroll
        for(int i=0;i<qk.num_elements;++i)qk_bmap|=static_cast<unsigned long long>(float(qk.x[i]))<<(i*8);
        #pragma unroll
        for(int i=0;i<pv.num_elements;++i)pv_bmap|=static_cast<unsigned long long>(float(pv.x[i]))<<(i*8);
    }
    __syncthreads();
    constexpr float scale_log2=float(double(0.08838834764831845f)*1.4426950408889634);
    for(int start=0;start<keytokens;start+=64) {
            if(warp<4) {
                wmma::fragment<wmma::accumulator,16,16,16,float> dots[2];
                #pragma unroll
                for(int sub=0;sub<2;++sub)wmma::fill_fragment(dots[sub],0.f);
                #pragma unroll
                for(int di=0;di<8;++di) {
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> b;
                    #pragma unroll
                    for(int i=0;i<b.num_elements;++i) {
                        const int m=(qk_bmap>>(i*8))&255,pos=start+warp*16+m/16,d=di*16+m%16;
                        const int gd=d;
                        b.x[i]=pos<keytokens?k[((int64_t(batch)*keytokens+pos)*hkv+kv)*128+gd]:T(0.f);
                    }
                    #pragma unroll
                    for(int sub=0;sub<2;++sub)wmma::mma_sync(dots[sub],qcache[sub][di],b,dots[sub]);
                }
                #pragma unroll
                for(int sub=0;sub<2;++sub)wmma::store_matrix_sync(scores+sub*16*NS+warp*16,dots[sub],NS,wmma::mem_row_major);
            }
        __syncthreads();
        if(tid<128) {
            int row=tid/4,lane=tid%4;
            int query=query0+row,begin=query/bs;
            float mx=-INFINITY;
            for(int j=lane;j<64;j+=4) {
                int pos=start+j;
                if(query>=tokens||pos<begin||pos>=keytokens) scores[row*NS+j]=-INFINITY;
                scores[row*NS+j]=__fmul_rn(__fmul_rn(scores[row*NS+j],0.08838834764831845f),1.4426950408889634f);
                mx=fmaxf(mx,scores[row*NS+j]);
            }
            for(int delta=2;delta;delta/=2) mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,delta,4));
            // A query can have an entirely masked leading/trailing tile.
            float nm=fmaxf(maximum[row],mx);
            float alpha=isfinite(nm)?exp2f(maximum[row]-nm):1.f;
            float sum=0;
            for(int j=lane;j<64;j+=4) {
                float prob=isfinite(nm)?exp2f(scores[row*NS+j]-nm):0.f;
                sum+=prob;ps[row*NS+j]=T(prob);
            }
            for(int delta=2;delta;delta/=2) sum+=__shfl_xor_sync(0xffffffff,sum,delta,4);
            if(lane==0) {denom[row]=denom[row]*alpha+sum;maximum[row]=nm;rescale[row]=alpha;}
        }
        __syncthreads();
        for(int i=tid;i<32*16;i+=256)scores[(i/16)*NS+i%16]=rescale[i/16];
        __syncthreads();
        if(warp<4)for(int sub=0;sub<2;++sub) {
            wmma::fragment<wmma::accumulator,16,16,16,float> factor;
            wmma::load_matrix_sync(factor,scores+sub*16*NS,NS,wmma::mem_row_major);
            for(int h=0;h<2;++h)for(int i=0;i<factor.num_elements;++i)output[sub*2+h].x[i]*=factor.x[i];
        }
        __syncthreads();
        if(warp<4) {
            #pragma unroll
            for(int half=0;half<2;++half) {
                #pragma unroll
                for(int j=0;j<64;j+=16) {
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> b;
                    #pragma unroll
                    for(int i=0;i<b.num_elements;++i) {
                        const int m=(pv_bmap>>(i*8))&255,pos=start+j+m/16,d=half*64+warp*16+m%16;
                        b.x[i]=pos<keytokens?v[((int64_t(batch)*keytokens+pos)*hkv+kv)*128+d]:T(0.f);
                    }
                    #pragma unroll
                    for(int sub=0;sub<2;++sub) {
                        wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                        wmma::load_matrix_sync(a,ps+sub*16*NS+j,NS);
                        wmma::mma_sync(output[sub*2+half],a,b,output[sub*2+half]);
                    }
                }
            }
        }
        __syncthreads();
    }
    for(int h=0;h<2;++h) {
        if(warp<4) for(int sub=0;sub<2;++sub)wmma::store_matrix_sync(scores+sub*16*NS+warp*16,output[sub*2+h],NS,wmma::mem_row_major);
        __syncthreads();
        for(int i=tid;i<32*64;i+=256) {
            int row=query0+i/64,d=h*64+i%64;
            if(row<tokens)out[((int64_t(batch)*tokens+row)*hq+head)*128+d]=T(scores[(i/64)*NS+i%64]/denom[i/64]);
        }
        __syncthreads();
    }
    if(lse && tid<32 && query0+tid<tokens) lse[(int64_t(batch)*hq+head)*tokens+query0+tid]=(maximum[tid]+log2f(denom[tid]))*0.6931471805599453f;
    return;
}
extern "C" int personal_nsa_branch_r49_cdkv32(const void* q,const void* k,const void* v,void* out,float* lse,
 int b,int t,int n,int hq,int hkv,int bs,int window,int kind,int dtype,void* stream) {
    if(kind==0&&dtype==1&&t>=1024&&hq/hkv==16) {

        nsaAttentionR49cdkv32Cuda<<<dim3((t+32-1)/32,b,hq),256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,t,n,hq,hkv,bs,lse);
        return int(cudaGetLastError());
    }
    return personal_nsa_branch_r47_creg(q,k,v,out,lse,b,t,n,hq,hkv,bs,window,kind,dtype,stream);
}
extern "C" int personal_nsa_resources_r49_cdkv32(int* values) {
    cudaFuncAttributes a;auto e=cudaFuncGetAttributes(&a,nsaAttentionR49cdkv32Cuda<__nv_bfloat16>);
    if(e!=cudaSuccess)return int(e);int blocks=0;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,nsaAttentionR49cdkv32Cuda<__nv_bfloat16>,256,0);
    values[0]=a.numRegs;values[1]=a.sharedSizeBytes;values[2]=a.localSizeBytes;values[3]=blocks;return int(e);
}
