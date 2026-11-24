#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#include <type_traits>
#include "compression_wmma.cuh"

__device__ float approxWindowP(float score, float maximum, float scale, float* pv) {
 float c=__uint_as_float(0x3e6e28f6u);
 float z=fminf(1.f,fmaxf(0.f,fmaf(score,scale*(1.f/64.f),fmaf(-maximum,scale*(1.f/64.f),c))));
 unsigned short b=(unsigned short)floorf(z*65535.f+0.5f);
 half a=__ushort_as_half(b),d=__ushort_as_half((unsigned short)(b+512));
 half r=__float2half_rn(fmaf(__half2float(a),__half2float(__ushort_as_half(0x3da8)),__half2float(d)));
 unsigned short bits=__half_as_ushort(r);
 *pv=__bfloat162float(__ushort_as_bfloat16((unsigned short)(((bits>>3)<0x80)?0:((bits>>3)+0x3800))));
 return __half2float(r);
}

#include "window_wmma.cuh"
template<class T> __global__ __launch_bounds__(128)
void nsaPrefillPoolCuda(const T* k,const T* v,T* kc,T* vc,int tokens,int kvheads,int blocks,int bs) {
    int dim=threadIdx.x,kv=blockIdx.y,block=blockIdx.x%blocks,batch=blockIdx.x/blocks;
    int start=block*bs,count=min(bs,tokens-start);
    int64_t off=((int64_t(batch)*tokens+start)*kvheads+kv)*128+dim;
    float sk=0,sv=0;
    for(int j=0;j<count;++j) {sk+=float(k[off]);sv+=float(v[off]);off+=kvheads*128;}
    int64_t out=((int64_t(batch)*blocks+block)*kvheads+kv)*128+dim;
    kc[out]=T(sk/count);vc[out]=T(sv/count);
    return;
}

// Compression: reverse-looking released mask, natural-log LSE in B,HQ,T.
// Window: causal raw-K/V span. Q is not rounded after scaling in these branches.
template<class T> __global__ __launch_bounds__(256)
void nsaPrefillBranchCuda(const T* q,const T* k,const T* v,T* out,float* lse,
                         int tokens,int keytokens,int hq,int hkv,int bs,int window,int kind,int approx) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    int group=hq/hkv,gh=blockIdx.z*8+warp,head=kv*group+gh;
    bool active=gh<group;
    int64_t qo=((int64_t(batch)*tokens+query)*hq+head)*128;
    float qd[4]={0,0,0,0},acc[4]={0,0,0,0};
    if(active) for(int d=0;d<4;++d) qd[d]=float(q[qo+lane+d*32]);
    int begin=kind==0?query/bs:(window>0?max(0,query-window+1):0);
    int end=kind==0?keytokens:query+1;
    float maximum=-INFINITY,denom=0;
    __shared__ float scores[8][64];
    int first=begin/64*64,last=(end-1)/64*64;
    int count=(last-first)/64+1;
    for(int iteration=0;iteration<count;++iteration) {
        // FlashAttention traverses key tiles from the query end. Compression
        // retains its own forward traversal; its released mask is unchanged.
        int start=kind==1?last-iteration*64:first+iteration*64;
        for(int j=0;j<64;++j) {
            int pos=start+j;float dot=0;
            if(active&&pos>=begin&&pos<end) {
                int64_t off=((int64_t(batch)*keytokens+pos)*hkv+kv)*128+lane;
                for(int d=0;d<4;++d) dot=fmaf(qd[d],float(k[off+d*32]),dot);
                for(int delta=16;delta;delta/=2) dot+=__shfl_down_sync(0xffffffff,dot,delta,32);
                if(kind==0) {dot=__fmul_rn(dot,0.08838834764831845f);dot=__fmul_rn(dot,1.4426950408889634f);}
            } else dot=-INFINITY;
            if(lane==0) scores[warp][j]=dot;
        }
        __syncwarp();
        float mx=fmaxf(scores[warp][lane],scores[warp][lane+32]);
        for(int delta=16;delta;delta/=2) mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,delta,32));
        constexpr float scale_log2=float(double(0.08838834764831845f)*1.4426950408889634);
        float nm=fmaxf(maximum,mx),alpha=kind==1?exp2f((maximum-nm)*scale_log2):exp2f(maximum-nm),sum=0;
        for(int j=lane;j<64;j+=32) {
            float p=active?(kind==1?exp2f(fmaf(scores[warp][j],scale_log2,-nm*scale_log2)):exp2f(scores[warp][j]-nm)):0.f;
            float pv=p;
            if(kind==1&&approx) { p=approxWindowP(scores[warp][j],nm,scale_log2,&pv); if(std::is_same<T,half>::value) pv=p; else p=pv; }
            else pv=float(T(p));
            sum+=p;scores[warp][j]=pv;
        }
        for(int delta=16;delta;delta/=2) sum+=__shfl_xor_sync(0xffffffff,sum,delta,32);
        denom=denom*alpha+sum;maximum=nm;
        for(int d=0;d<4;++d) acc[d]*=alpha;
        __syncwarp();
        if(active) for(int j=max(0,begin-start);j<64&&start+j<end;++j) {
            float p=scores[warp][j];
            int64_t off=((int64_t(batch)*keytokens+start+j)*hkv+kv)*128+lane;
            for(int d=0;d<4;++d) acc[d]=fmaf(p,float(v[off+d*32]),acc[d]);
        }
        __syncwarp();
    }
    if(active) {
        for(int d=0;d<4;++d) out[qo+lane+d*32]=T(acc[d]/denom);
        if(lse&&lane==0) lse[(int64_t(batch)*hq+head)*tokens+query]=maximum*0.6931471805599453f+logf(denom);
    }
    return;
}

// A volatile bit representation makes the low-precision materialization
// observable even when the compatibility compiler retains excess precision.
template<class G> __device__ float gate_materialize(float x);
template<> __device__ float gate_materialize<half>(float x) {
    volatile unsigned short bits=__half_as_ushort(__float2half_rn(x));
    return __half2float(__ushort_as_half(bits));
}
template<> __device__ float gate_materialize<__nv_bfloat16>(float x) {
    volatile unsigned short bits=__bfloat16_as_ushort(__float2bfloat16_rn(x));
    return __bfloat162float(__ushort_as_bfloat16(bits));
}
template<> __device__ float gate_materialize<float>(float x) {
    volatile float result=x;
    return result;
}

template<class T,class G> __global__
void nsaPrefillGateCuda(const T* oc,const T* os,const T* ow,const G* gc,const G* gs,const G* gw,G* out,int64_t elements) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<elements) {
        int64_t row=i/128;
        // Match eager Torch multiplication/addition materialization boundaries.
        float pc=gate_materialize<G>(__fmul_rn(float(oc[i]),float(gc[row])));
        float ps=gate_materialize<G>(__fmul_rn(float(os[i]),float(gs[row])));
        float pw=gate_materialize<G>(__fmul_rn(float(ow[i]),float(gw[row])));
        float partial=gate_materialize<G>(__fadd_rn(pc,ps));
        out[i]=G(__fadd_rn(partial,pw));
    }
    return;
}

extern "C" int personal_nsa_pool(const void* k,const void* v,void* kc,void* vc,int b,int t,int hkv,int bs,int dtype,void* stream) {
    int n=(t+bs-1)/bs;dim3 grid(b*n,hkv);
    if(dtype==0) nsaPrefillPoolCuda<<<grid,128,0,(cudaStream_t)stream>>>((const half*)k,(const half*)v,(half*)kc,(half*)vc,t,hkv,n,bs);
    else nsaPrefillPoolCuda<<<grid,128,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)kc,(__nv_bfloat16*)vc,t,hkv,n,bs);
    return int(cudaGetLastError());
}
extern "C" int personal_nsa_branch(const void* q,const void* k,const void* v,void* out,float* lse,
 int b,int t,int n,int hq,int hkv,int bs,int window,int kind,int dtype,void* stream) {
    if(kind==0) {
      dim3 cg((t+15)/16,b,hq);
      if(dtype==0)nsaCompressionWmmaCuda<<<cg,256,0,(cudaStream_t)stream>>>((const half*)q,(const half*)k,(const half*)v,(half*)out,t,n,hq,hkv,bs,lse);
      else nsaCompressionWmmaCuda<<<cg,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,t,n,hq,hkv,bs,lse);
      return int(cudaGetLastError());
    }
    int dev=0,sm=0;cudaError_t e=cudaGetDevice(&dev);if(e!=cudaSuccess)return int(e);
    e=cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,dev);if(e!=cudaSuccess)return int(e);
    int approx=t>=1024&&int64_t((t+63)/64)*b*hq>=int64_t(sm)*4;
    if(kind==1 && approx && dtype==1) {
      dim3 wg((t+15)/16,b,hq);
      if(dtype==0)nsaPrefillWindowRegisterCuda<<<wg,256,0,(cudaStream_t)stream>>>((const half*)q,(const half*)k,(const half*)v,(half*)out,t,hq,hkv,window);
      else nsaPrefillWindowRegisterCuda<<<wg,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,t,hq,hkv,window);
      return int(cudaGetLastError());
    }
    dim3 grid(b*t,hkv,(hq/hkv+7)/8);
    if(dtype==0) nsaPrefillBranchCuda<<<grid,256,0,(cudaStream_t)stream>>>((const half*)q,(const half*)k,(const half*)v,(half*)out,lse,t,n,hq,hkv,bs,window,kind,approx);
    else nsaPrefillBranchCuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,lse,t,n,hq,hkv,bs,window,kind,approx);
    return int(cudaGetLastError());
}
extern "C" int personal_nsa_gate(const void* oc,const void* os,const void* ow,const void* gc,const void* gs,const void* gw,void* out,
 int64_t elements,int dtype,int gate32,void* stream) {
    dim3 grid((elements+255)/256);
    if(dtype==0&&gate32==0) nsaPrefillGateCuda<<<grid,256,0,(cudaStream_t)stream>>>((const half*)oc,(const half*)os,(const half*)ow,(const half*)gc,(const half*)gs,(const half*)gw,(half*)out,elements);
    else if(dtype==1&&gate32==0) nsaPrefillGateCuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)oc,(const __nv_bfloat16*)os,(const __nv_bfloat16*)ow,(const __nv_bfloat16*)gc,(const __nv_bfloat16*)gs,(const __nv_bfloat16*)gw,(__nv_bfloat16*)out,elements);
    else if(dtype==0) nsaPrefillGateCuda<<<grid,256,0,(cudaStream_t)stream>>>((const half*)oc,(const half*)os,(const half*)ow,(const float*)gc,(const float*)gs,(const float*)gw,(float*)out,elements);
    else nsaPrefillGateCuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)oc,(const __nv_bfloat16*)os,(const __nv_bfloat16*)ow,(const float*)gc,(const float*)gs,(const float*)gw,(float*)out,elements);
    return int(cudaGetLastError());
}

template<class T> __global__ __launch_bounds__(256)
void nsaCompressionRegisterR47Cuda(const T* q,const T* k,const T* v,T* out,
                             int tokens,int keytokens,int hq,int hkv,int bs,float* lse) {
    using namespace nvcuda;
    constexpr int DS=136,NS=72; // Padding avoids power-of-two shared-memory strides.
    int tid=threadIdx.x,warp=tid/warpSize,query0=blockIdx.x*16,batch=blockIdx.y,head=blockIdx.z;
    int kv=head/(hq/hkv);
    __shared__ __align__(32) T qs[16*DS],tiles[64*DS],ps[16*NS];
    T* ks=tiles;T* vs=tiles; // K is no longer needed when P*V begins.
    __shared__ __align__(32) float scores[16*NS];
    __shared__ float maximum[16],denom[16],rescale[16];
    for(int i=tid;i<16*128;i+=256) {
        int row=query0+i/128,d=i%128;
        qs[(i/128)*DS+d]=row<tokens?q[((int64_t(batch)*tokens+row)*hq+head)*128+d]:T(0.f);
    }
    if(tid<16) {maximum[tid]=-INFINITY;denom[tid]=0;}
    __syncthreads();
    wmma::fragment<wmma::accumulator,16,16,16,float> output[2];
    if(warp<4)for(int h=0;h<2;++h)wmma::fill_fragment(output[h],0.f);
    constexpr float scale_log2=float(double(0.08838834764831845f)*1.4426950408889634);
    for(int start=0;start<keytokens;start+=64) {
        for(int i=tid;i<64*128;i+=256) {
            int pos=start+i/128,d=i%128;
            int64_t off=((int64_t(batch)*keytokens+pos)*hkv+kv)*128+d;
            ks[(i/128)*DS+d]=pos<keytokens?k[off]:T(0.f);
        }
        __syncthreads();
        if(warp<4) {
            for(int col=warp*16;col<64;col+=64) {
                wmma::fragment<wmma::accumulator,16,16,16,float> c;
                wmma::fill_fragment(c,0.f);
                for(int d=0;d<128;d+=16) {
                    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> b;
                    wmma::load_matrix_sync(a,qs+d,DS);
                    wmma::load_matrix_sync(b,ks+col*DS+d,DS);
                    wmma::mma_sync(c,a,b,c);
                }
                wmma::store_matrix_sync(scores+col,c,NS,wmma::mem_row_major);
            }
        }
        __syncthreads();
        for(int i=tid;i<64*128;i+=256) {
            int pos=start+i/128,d=i%128;
            int64_t off=((int64_t(batch)*keytokens+pos)*hkv+kv)*128+d;
            vs[(i/128)*DS+d]=pos<keytokens?v[off]:T(0.f);
        }
        __syncthreads();
        if(tid<64) {
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
        for(int i=tid;i<16*16;i+=256)scores[(i/16)*NS+i%16]=rescale[i/16];
        __syncthreads();
        if(warp<4) {
            wmma::fragment<wmma::accumulator,16,16,16,float> factor;
            wmma::load_matrix_sync(factor,scores,NS,wmma::mem_row_major);
            for(int h=0;h<2;++h)for(int i=0;i<factor.num_elements;++i)output[h].x[i]*=factor.x[i];
        }
        __syncthreads();
        if(warp<4) {
            for(int col=warp*16;col<128;col+=64) {
                auto &c=output[col/64];
                for(int j=0;j<64;j+=16) {
                    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> b;
                    wmma::load_matrix_sync(a,ps+j,NS);
                    wmma::load_matrix_sync(b,vs+j*DS+col,DS);
                    wmma::mma_sync(c,a,b,c);
                }
            }
        }
        __syncthreads();
    }
    for(int h=0;h<2;++h) {
        if(warp<4)wmma::store_matrix_sync(scores+warp*16,output[h],NS,wmma::mem_row_major);
        __syncthreads();
        for(int i=tid;i<16*64;i+=256) {
            int row=query0+i/64,d=h*64+i%64;
            if(row<tokens)out[((int64_t(batch)*tokens+row)*hq+head)*128+d]=T(scores[(i/64)*NS+i%64]/denom[i/64]);
        }
        __syncthreads();
    }
    if(lse && tid<16 && query0+tid<tokens) lse[(int64_t(batch)*hq+head)*tokens+query0+tid]=(maximum[tid]+log2f(denom[tid]))*0.6931471805599453f;
    return;
}

template<class T> __global__ __launch_bounds__(256)
void nsaWindowR47SoftCuda(const T* q,const T* k,const T* v,T* out,
                             int tokens,int hq,int hkv,int window) {
    using namespace nvcuda;
    constexpr int DS=136,NS=72,KS=72; // Padding avoids power-of-two shared-memory strides.
    int tid=threadIdx.x,warp=tid/warpSize,query0=blockIdx.x*16,batch=blockIdx.y,head=blockIdx.z;
    int kv=head/(hq/hkv);
    __shared__ __align__(32) T qs[16*DS],tiles[64*KS],ps[16*NS];
    T* ks=tiles;T* vs=tiles; // K is no longer needed when P*V begins.
    __shared__ __align__(32) float scores[16*NS];
    __shared__ float maximum[16],denom[16],rescale[16];
    for(int i=tid;i<16*128;i+=256) {
        int row=query0+i/128,d=i%128;int gd=d;gd=(d/32)*32+((d%16)/4)*8+d%4+((d%32)/16)*4;
        qs[(i/128)*DS+d]=row<tokens?q[((int64_t(batch)*tokens+row)*hq+head)*128+gd]:T(0.f);

    }
    if(tid<16) {maximum[tid]=-INFINITY;denom[tid]=0;}
    __syncthreads();
    wmma::fragment<wmma::accumulator,16,16,16,float> output[2];
    if(warp<4)for(int h=0;h<2;++h)wmma::fill_fragment(output[h],0.f);
    int earliest=window>0?max(0,(query0/128)*128-window+1):0;
    int latest=min(tokens-1,(query0/128)*128+127);
    constexpr float scale_log2=float(double(0.08838834764831845f)*1.4426950408889634);
    for(int start=latest/64*64;start>=earliest/64*64;start-=64) {
        const int query_end=min(tokens-1,query0+15);
        const int query_begin=window>0?max(0,query0-window+1):0;
        const bool empty=start>query_end || start+63<query_begin;
        if(!empty) {
            wmma::fragment<wmma::accumulator,16,16,16,float> c;
            wmma::fill_fragment(c,0.f);
            for(int half=0;half<2;++half) {
                for(int i=tid;i<64*64;i+=256) {
                    int pos=start+i/64,d=half*64+i%64;
                    int gd=(d/32)*32+((d%16)/4)*8+d%4+((d%32)/16)*4;
                    ks[(i/64)*KS+i%64]=pos<tokens?k[((int64_t(batch)*tokens+pos)*hkv+kv)*128+gd]:T(0.f);
                }
                __syncthreads();
                if(warp<4)for(int di=0;di<4;++di) {
                    int d=di*16;
                    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> b;
                    wmma::load_matrix_sync(a,qs+half*64+d,DS);
                    wmma::load_matrix_sync(b,ks+warp*16*KS+d,KS);
                    wmma::mma_sync(c,a,b,c);
                }
                __syncthreads();
            }
            if(warp<4)wmma::store_matrix_sync(scores+warp*16,c,NS,wmma::mem_row_major);
        } else {
            for(int i=tid;i<16*64;i+=256)scores[(i/64)*NS+i%64]=-INFINITY;
        }
        __syncthreads();
        {
            const int row=tid/16,lane=tid%16,query=query0+row;
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
        if(tid<64) {
            const int row=tid/4,lane=tid%4;
            float lo=0.f,hi=0.f;
            for(int tile=0;tile<4;++tile)for(int pair=0;pair<2;++pair){const int j=tile*16+lane*4+pair*2;lo=halfRound(lo+scores[row*NS+j]);hi=halfRound(hi+scores[row*NS+j+1]);}
            float sum=halfRound(lo+hi);
            for(int delta=2;delta;delta/=2)sum+=__shfl_xor_sync(0xffffffff,sum,delta,4);
            if(lane==0)denom[row]=denom[row]*rescale[row]+sum;
        }
        __syncthreads();
        for(int i=tid;i<16*16;i+=256)scores[(i/16)*NS+i%16]=rescale[i/16];
        __syncthreads();
        if(warp<4){
            wmma::fragment<wmma::accumulator,16,16,16,float> factor;
            wmma::load_matrix_sync(factor,scores,NS,wmma::mem_row_major);
            for(int h=0;h<2;++h)for(int i=0;i<factor.num_elements;++i)output[h].x[i]*=factor.x[i];
        }
        __syncthreads();
        if(!empty) {
        for(int half=0;half<2;++half) {
            for(int i=tid;i<64*64;i+=256) {
                int pos=start+i/64,d=half*64+i%64;
                vs[(i/64)*KS+i%64]=pos<tokens?v[((int64_t(batch)*tokens+pos)*hkv+kv)*128+d]:T(0.f);
            }
            __syncthreads();
            if(warp<4) {
                const int col=half*64+warp*16;
                auto &c=output[half];
                for(int j=0;j<64;j+=16) {
                    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> b;
                    wmma::load_matrix_sync(a,ps+j,NS);
                    wmma::load_matrix_sync(b,vs+j*KS+warp*16,KS);
                    wmma::mma_sync(c,a,b,c);
                }

            }
            __syncthreads();
        }
        }
    }
    for(int h=0;h<2;++h) {
        if(warp<4)wmma::store_matrix_sync(scores+warp*16,output[h],NS,wmma::mem_row_major);
        __syncthreads();
        for(int i=tid;i<16*64;i+=256) {
            int row=query0+i/64,d=h*64+i%64;
            if(row<tokens)out[((int64_t(batch)*tokens+row)*hq+head)*128+d]=T(__bfloat162float(__ushort_as_bfloat16((unsigned short)((__float_as_uint(scores[(i/64)*NS+i%64]/denom[i/64]*1.00390625f)+0x8000u)>>16))));
        }
        __syncthreads();
    }
    return;
}

template<class T,class G> __global__ __launch_bounds__(256)
void nsaWindowR47GateCuda(const T* q,const T* k,const T* v,G* out,const T* oc,const T* os,const G* gc,const G* gs,const G* gw,
                             int tokens,int hq,int hkv,int window) {
    using namespace nvcuda;
    constexpr int DS=136,NS=72,KS=72; // Padding avoids power-of-two shared-memory strides.
    int tid=threadIdx.x,warp=tid/warpSize,query0=blockIdx.x*16,batch=blockIdx.y,head=blockIdx.z;
    int kv=head/(hq/hkv);
    __shared__ __align__(32) T qs[16*DS],tiles[64*KS],ps[16*NS];
    T* ks=tiles;T* vs=tiles; // K is no longer needed when P*V begins.
    __shared__ __align__(32) float scores[16*NS];
    __shared__ float maximum[16],denom[16],rescale[16];
    for(int i=tid;i<16*128;i+=256) {
        int row=query0+i/128,d=i%128;int gd=d;gd=(d/32)*32+((d%16)/4)*8+d%4+((d%32)/16)*4;
        qs[(i/128)*DS+d]=row<tokens?q[((int64_t(batch)*tokens+row)*hq+head)*128+gd]:T(0.f);

    }
    if(tid<16) {maximum[tid]=-INFINITY;denom[tid]=0;}
    __syncthreads();
    wmma::fragment<wmma::accumulator,16,16,16,float> output[2];
    if(warp<4)for(int h=0;h<2;++h)wmma::fill_fragment(output[h],0.f);
    int earliest=window>0?max(0,(query0/128)*128-window+1):0;
    int latest=min(tokens-1,(query0/128)*128+127);
    constexpr float scale_log2=float(double(0.08838834764831845f)*1.4426950408889634);
    for(int start=latest/64*64;start>=earliest/64*64;start-=64) {
        const int query_end=min(tokens-1,query0+15);
        const int query_begin=window>0?max(0,query0-window+1):0;
        const bool empty=start>query_end || start+63<query_begin;
        if(!empty) {
            wmma::fragment<wmma::accumulator,16,16,16,float> c;
            wmma::fill_fragment(c,0.f);
            for(int half=0;half<2;++half) {
                for(int i=tid;i<64*64;i+=256) {
                    int pos=start+i/64,d=half*64+i%64;
                    int gd=(d/32)*32+((d%16)/4)*8+d%4+((d%32)/16)*4;
                    ks[(i/64)*KS+i%64]=pos<tokens?k[((int64_t(batch)*tokens+pos)*hkv+kv)*128+gd]:T(0.f);
                }
                __syncthreads();
                if(warp<4)for(int di=0;di<4;++di) {
                    int d=di*16;
                    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> b;
                    wmma::load_matrix_sync(a,qs+half*64+d,DS);
                    wmma::load_matrix_sync(b,ks+warp*16*KS+d,KS);
                    wmma::mma_sync(c,a,b,c);
                }
                __syncthreads();
            }
            if(warp<4)wmma::store_matrix_sync(scores+warp*16,c,NS,wmma::mem_row_major);
        } else {
            for(int i=tid;i<16*64;i+=256)scores[(i/64)*NS+i%64]=-INFINITY;
        }
        __syncthreads();
        if(tid<64) {
            int row=tid/4,lane=tid%4;
            int query=query0+row,begin=window>0?max(0,query-window+1):0;
            float mx=-INFINITY;
            for(int j=lane;j<64;j+=4) {
                int pos=start+j;
                if(query>=tokens||pos<begin||pos>query) scores[row*NS+j]=-INFINITY;
                mx=fmaxf(mx,scores[row*NS+j]);
            }
            for(int delta=2;delta;delta/=2) mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,delta,4));
            // A query can have an entirely masked leading/trailing tile.
            float nm=fmaxf(maximum[row],mx);
            float alpha=isfinite(nm)?exp2f((maximum[row]-nm)*scale_log2):1.f;
            float sum=0;
            for(int j=lane;j<64;j+=4) {
                float prob=0.f,pv=0.f; if(isfinite(nm)) {prob=approxWindowP(scores[row*NS+j],nm,scale_log2,&pv);if(std::is_same<T,half>::value)pv=prob;}
                scores[row*NS+j]=prob;ps[row*NS+j]=T(pv);
            }
            __syncwarp();
            float lo=0.f,hi=0.f;
            for(int tile=0;tile<4;++tile)for(int pair=0;pair<2;++pair){int j=tile*16+lane*4+pair*2;lo=halfRound(lo+scores[row*NS+j]);hi=halfRound(hi+scores[row*NS+j+1]);}
            sum=halfRound(lo+hi);
            for(int delta=2;delta;delta/=2) sum+=__shfl_xor_sync(0xffffffff,sum,delta,4);
            if(lane==0) {denom[row]=denom[row]*alpha+sum;maximum[row]=nm;rescale[row]=alpha;}
        }
        __syncthreads();
        for(int i=tid;i<16*16;i+=256)scores[(i/16)*NS+i%16]=rescale[i/16];
        __syncthreads();
        if(warp<4){
            wmma::fragment<wmma::accumulator,16,16,16,float> factor;
            wmma::load_matrix_sync(factor,scores,NS,wmma::mem_row_major);
            for(int h=0;h<2;++h)for(int i=0;i<factor.num_elements;++i)output[h].x[i]*=factor.x[i];
        }
        __syncthreads();
        if(!empty) {
        for(int half=0;half<2;++half) {
            for(int i=tid;i<64*64;i+=256) {
                int pos=start+i/64,d=half*64+i%64;
                vs[(i/64)*KS+i%64]=pos<tokens?v[((int64_t(batch)*tokens+pos)*hkv+kv)*128+d]:T(0.f);
            }
            __syncthreads();
            if(warp<4) {
                const int col=half*64+warp*16;
                auto &c=output[half];
                for(int j=0;j<64;j+=16) {
                    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> b;
                    wmma::load_matrix_sync(a,ps+j,NS);
                    wmma::load_matrix_sync(b,vs+j*KS+warp*16,KS);
                    wmma::mma_sync(c,a,b,c);
                }

            }
            __syncthreads();
        }
        }
    }
    for(int h=0;h<2;++h) {
        if(warp<4)wmma::store_matrix_sync(scores+warp*16,output[h],NS,wmma::mem_row_major);
        __syncthreads();
        for(int i=tid;i<16*64;i+=256) {
            int row=query0+i/64,d=h*64+i%64;
            if(row<tokens) {
                const int64_t headrow=(int64_t(batch)*tokens+row)*hq+head;
                const int64_t index=headrow*128+d;
                const T ow=T(__bfloat162float(__ushort_as_bfloat16((unsigned short)((__float_as_uint(scores[(i/64)*NS+i%64]/denom[i/64]*1.00390625f)+0x8000u)>>16))));
                const float pc=gate_materialize<G>(__fmul_rn(float(oc[index]),float(gc[headrow])));
                const float ps_=gate_materialize<G>(__fmul_rn(float(os[index]),float(gs[headrow])));
                const float pw=gate_materialize<G>(__fmul_rn(float(ow),float(gw[headrow])));
                const float partial=gate_materialize<G>(__fadd_rn(pc,ps_));
                out[index]=G(__fadd_rn(partial,pw));
            }
        }
        __syncthreads();
    }
    return;
}

template<class T,class G> __global__ __launch_bounds__(256)
void nsaWindowR47SoftGateCuda(const T* q,const T* k,const T* v,G* out,const T* oc,const T* os,const G* gc,const G* gs,const G* gw,
                             int tokens,int hq,int hkv,int window) {
    using namespace nvcuda;
    constexpr int DS=136,NS=72,KS=72; // Padding avoids power-of-two shared-memory strides.
    int tid=threadIdx.x,warp=tid/warpSize,query0=blockIdx.x*16,batch=blockIdx.y,head=blockIdx.z;
    int kv=head/(hq/hkv);
    __shared__ __align__(32) T qs[16*DS],tiles[64*KS],ps[16*NS];
    T* ks=tiles;T* vs=tiles; // K is no longer needed when P*V begins.
    __shared__ __align__(32) float scores[16*NS];
    __shared__ float maximum[16],denom[16],rescale[16];
    for(int i=tid;i<16*128;i+=256) {
        int row=query0+i/128,d=i%128;int gd=d;gd=(d/32)*32+((d%16)/4)*8+d%4+((d%32)/16)*4;
        qs[(i/128)*DS+d]=row<tokens?q[((int64_t(batch)*tokens+row)*hq+head)*128+gd]:T(0.f);

    }
    if(tid<16) {maximum[tid]=-INFINITY;denom[tid]=0;}
    __syncthreads();
    wmma::fragment<wmma::accumulator,16,16,16,float> output[2];
    if(warp<4)for(int h=0;h<2;++h)wmma::fill_fragment(output[h],0.f);
    int earliest=window>0?max(0,(query0/128)*128-window+1):0;
    int latest=min(tokens-1,(query0/128)*128+127);
    constexpr float scale_log2=float(double(0.08838834764831845f)*1.4426950408889634);
    for(int start=latest/64*64;start>=earliest/64*64;start-=64) {
        const int query_end=min(tokens-1,query0+15);
        const int query_begin=window>0?max(0,query0-window+1):0;
        const bool empty=start>query_end || start+63<query_begin;
        if(!empty) {
            wmma::fragment<wmma::accumulator,16,16,16,float> c;
            wmma::fill_fragment(c,0.f);
            for(int half=0;half<2;++half) {
                for(int i=tid;i<64*64;i+=256) {
                    int pos=start+i/64,d=half*64+i%64;
                    int gd=(d/32)*32+((d%16)/4)*8+d%4+((d%32)/16)*4;
                    ks[(i/64)*KS+i%64]=pos<tokens?k[((int64_t(batch)*tokens+pos)*hkv+kv)*128+gd]:T(0.f);
                }
                __syncthreads();
                if(warp<4)for(int di=0;di<4;++di) {
                    int d=di*16;
                    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::col_major> b;
                    wmma::load_matrix_sync(a,qs+half*64+d,DS);
                    wmma::load_matrix_sync(b,ks+warp*16*KS+d,KS);
                    wmma::mma_sync(c,a,b,c);
                }
                __syncthreads();
            }
            if(warp<4)wmma::store_matrix_sync(scores+warp*16,c,NS,wmma::mem_row_major);
        } else {
            for(int i=tid;i<16*64;i+=256)scores[(i/64)*NS+i%64]=-INFINITY;
        }
        __syncthreads();
        {
            const int row=tid/16,lane=tid%16,query=query0+row;
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
        if(tid<64) {
            const int row=tid/4,lane=tid%4;
            float lo=0.f,hi=0.f;
            for(int tile=0;tile<4;++tile)for(int pair=0;pair<2;++pair){const int j=tile*16+lane*4+pair*2;lo=halfRound(lo+scores[row*NS+j]);hi=halfRound(hi+scores[row*NS+j+1]);}
            float sum=halfRound(lo+hi);
            for(int delta=2;delta;delta/=2)sum+=__shfl_xor_sync(0xffffffff,sum,delta,4);
            if(lane==0)denom[row]=denom[row]*rescale[row]+sum;
        }
        __syncthreads();
        for(int i=tid;i<16*16;i+=256)scores[(i/16)*NS+i%16]=rescale[i/16];
        __syncthreads();
        if(warp<4){
            wmma::fragment<wmma::accumulator,16,16,16,float> factor;
            wmma::load_matrix_sync(factor,scores,NS,wmma::mem_row_major);
            for(int h=0;h<2;++h)for(int i=0;i<factor.num_elements;++i)output[h].x[i]*=factor.x[i];
        }
        __syncthreads();
        if(!empty) {
        for(int half=0;half<2;++half) {
            for(int i=tid;i<64*64;i+=256) {
                int pos=start+i/64,d=half*64+i%64;
                vs[(i/64)*KS+i%64]=pos<tokens?v[((int64_t(batch)*tokens+pos)*hkv+kv)*128+d]:T(0.f);
            }
            __syncthreads();
            if(warp<4) {
                const int col=half*64+warp*16;
                auto &c=output[half];
                for(int j=0;j<64;j+=16) {
                    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> b;
                    wmma::load_matrix_sync(a,ps+j,NS);
                    wmma::load_matrix_sync(b,vs+j*KS+warp*16,KS);
                    wmma::mma_sync(c,a,b,c);
                }

            }
            __syncthreads();
        }
        }
    }
    for(int h=0;h<2;++h) {
        if(warp<4)wmma::store_matrix_sync(scores+warp*16,output[h],NS,wmma::mem_row_major);
        __syncthreads();
        for(int i=tid;i<16*64;i+=256) {
            int row=query0+i/64,d=h*64+i%64;
            if(row<tokens) {
                const int64_t headrow=(int64_t(batch)*tokens+row)*hq+head;
                const int64_t index=headrow*128+d;
                const T ow=T(__bfloat162float(__ushort_as_bfloat16((unsigned short)((__float_as_uint(scores[(i/64)*NS+i%64]/denom[i/64]*1.00390625f)+0x8000u)>>16))));
                const float pc=gate_materialize<G>(__fmul_rn(float(oc[index]),float(gc[headrow])));
                const float ps_=gate_materialize<G>(__fmul_rn(float(os[index]),float(gs[headrow])));
                const float pw=gate_materialize<G>(__fmul_rn(float(ow),float(gw[headrow])));
                const float partial=gate_materialize<G>(__fadd_rn(pc,ps_));
                out[index]=G(__fadd_rn(partial,pw));
            }
        }
        __syncthreads();
    }
    return;
}

static int r47_window_fast(int b,int t,int hq,int dtype,int *fast) {
    if(dtype!=1||t<1024){*fast=0;return 0;}
    int device=0,sm=0;auto e=cudaGetDevice(&device);if(e!=cudaSuccess)return int(e);
    e=cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device);if(e!=cudaSuccess)return int(e);
    *fast=dtype==1&&t>=1024&&int64_t((t+63)/64)*b*hq>=int64_t(sm)*4;return 0;
}
extern "C" int personal_nsa_branch_r47_creg(const void* q,const void* k,const void* v,void* out,float* lse,
 int b,int t,int n,int hq,int hkv,int bs,int window,int kind,int dtype,void* stream) {
    if(kind!=0)return personal_nsa_branch(q,k,v,out,lse,b,t,n,hq,hkv,bs,window,kind,dtype,stream);
    dim3 grid((t+15)/16,b,hq);
    if(dtype==0)nsaCompressionRegisterR47Cuda<<<grid,256,0,(cudaStream_t)stream>>>((const half*)q,(const half*)k,(const half*)v,(half*)out,t,n,hq,hkv,bs,lse);
    else nsaCompressionRegisterR47Cuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,t,n,hq,hkv,bs,lse);
    return int(cudaGetLastError());
}
extern "C" int personal_nsa_branch_r47_wsoft(const void* q,const void* k,const void* v,void* out,float* lse,
 int b,int t,int n,int hq,int hkv,int bs,int window,int kind,int dtype,void* stream) {
    if(kind!=1)return personal_nsa_branch(q,k,v,out,lse,b,t,n,hq,hkv,bs,window,kind,dtype,stream);
    int fast=0;int code=r47_window_fast(b,t,hq,dtype,&fast);if(code)return code;
    if(!fast)return personal_nsa_branch(q,k,v,out,lse,b,t,n,hq,hkv,bs,window,kind,dtype,stream);
    dim3 grid((t+15)/16,b,hq);
    nsaWindowR47SoftCuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,t,hq,hkv,window);
    return int(cudaGetLastError());
}

extern "C" int personal_nsa_window_gate_r47_wgate(const void* q,const void* k,const void* v,
 const void* oc,const void* os,void* ow,const void* gc,const void* gs,const void* gw,void* out,
 int b,int t,int hq,int hkv,int bs,int window,int dtype,int gate32,void* stream) {
    int fast=0;int code=r47_window_fast(b,t,hq,dtype,&fast);if(code)return code;
    if(!fast){code=personal_nsa_branch(q,k,v,ow,nullptr,b,t,t,hq,hkv,bs,window,1,dtype,stream);if(code)return code;
        return personal_nsa_gate(oc,os,ow,gc,gs,gw,out,int64_t(b)*t*hq*128,dtype,gate32,stream);}
    dim3 grid((t+15)/16,b,hq);
    if(gate32)nsaWindowR47GateCuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(float*)out,
        (const __nv_bfloat16*)oc,(const __nv_bfloat16*)os,(const float*)gc,(const float*)gs,(const float*)gw,t,hq,hkv,window);
    else nsaWindowR47GateCuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,
        (const __nv_bfloat16*)oc,(const __nv_bfloat16*)os,(const __nv_bfloat16*)gc,(const __nv_bfloat16*)gs,(const __nv_bfloat16*)gw,t,hq,hkv,window);
    return int(cudaGetLastError());
}

extern "C" int personal_nsa_window_gate_r47_wsoftgate(const void* q,const void* k,const void* v,
 const void* oc,const void* os,void* ow,const void* gc,const void* gs,const void* gw,void* out,
 int b,int t,int hq,int hkv,int bs,int window,int dtype,int gate32,void* stream) {
    int fast=0;int code=r47_window_fast(b,t,hq,dtype,&fast);if(code)return code;
    if(!fast){code=personal_nsa_branch(q,k,v,ow,nullptr,b,t,t,hq,hkv,bs,window,1,dtype,stream);if(code)return code;
        return personal_nsa_gate(oc,os,ow,gc,gs,gw,out,int64_t(b)*t*hq*128,dtype,gate32,stream);}
    dim3 grid((t+15)/16,b,hq);
    if(gate32)nsaWindowR47SoftGateCuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(float*)out,
        (const __nv_bfloat16*)oc,(const __nv_bfloat16*)os,(const float*)gc,(const float*)gs,(const float*)gw,t,hq,hkv,window);
    else nsaWindowR47SoftGateCuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)out,
        (const __nv_bfloat16*)oc,(const __nv_bfloat16*)os,(const __nv_bfloat16*)gc,(const __nv_bfloat16*)gs,(const __nv_bfloat16*)gw,t,hq,hkv,window);
    return int(cudaGetLastError());
}
