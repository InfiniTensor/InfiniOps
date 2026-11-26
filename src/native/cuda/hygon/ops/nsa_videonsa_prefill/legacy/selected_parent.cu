#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

// Independent 32-thread CUDA subgroups, one query head/subgroup. No K/V LDS
// tiles or block-wide barriers; only scores/probabilities occupy shared memory.
template<class T> __global__ __launch_bounds__(256)
void nsaPrefillSelectedWarpCuda(const T* q,const T* k,const T* v,const int32_t* ids,
                              T* out,int tokens,int hq,int hkv,int width,int bs,float scale) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    int group=hq/hkv,gh=blockIdx.z*8+warp,head=kv*group+gh;
    bool active=gh<group;
    int64_t qo=((int64_t(batch)*tokens+query)*hq+head)*128;
    int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ float scores[8][128];
    __shared__ float rounded_q[128][9];
    float qd[4]={0,0,0,0},acc[4]={0,0,0,0};
    if(active) for(int d=0;d<4;++d) qd[d]=float(T(float(q[qo+lane+d*32])*scale));
    // G7 preserves the author's dimension-ordered FMA, but assigns
    // independent token dots to lanes. Only Q and scores use shared memory.
    if((group==7 || group==16) && active)
        for(int d=lane;d<128;d+=32)
            rounded_q[d][warp]=float(T(float(q[qo+d])*scale));
    if(group==16)__syncthreads();else __syncwarp();
    float maximum=-INFINITY,denom=0;
    for(int slot=0;slot<width;++slot) {
        int id=ids[io+slot];
        if(id<0||int64_t(id)*bs>query) continue;
        int start=id*bs;
        int valid=min(bs,query+1-start);
        if(group==16)__syncthreads();
        if(group==7) {
            for(int j=lane;j<bs;j+=32) {
                float dot=-INFINITY;
                if(active && j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    dot=0.f;
                    for(int d=0;d<128;++d)
                        dot=fmaf(rounded_q[d][warp],float(k[kb+d]),dot);
                }
                scores[warp][j]=dot;
            }
        } else if(group==16) {
            for(int task=threadIdx.x;task<8*bs;task+=blockDim.x) {
                const int lh=task%8,j=task/8;
                float dot=-INFINITY;
                if(j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    float parts[32]={};
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        #pragma unroll
                        for(int l=0;l<32;++l)
                            parts[l]=fmaf(rounded_q[l+i*32][lh],float(k[kb+l+i*32]),parts[l]);
                    }
                    #pragma unroll
                    for(int delta=16;delta;delta/=2) {
                        #pragma unroll
                        for(int l=0;l<delta;++l)parts[l]+=parts[l+delta];
                    }
                    dot=parts[0];
                }
                scores[lh][j]=dot;
            }
        } else {
        // 2 independent token dots in flight; each retains its original
        // dimension FMA and shuffle reduction order. No additional K/V LDS.
        for(int base=0;base<bs;base+=2) {
            float score[2]={0};
            #pragma unroll
            for(int u=0;u<2;++u) {
                int j=base+u;
                if(active&&j<valid) {
                    int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane;
                    if(group==7) {
                      if(lane==0)for(int d=0;d<128;++d)score[u]=fmaf(float(T(float(q[qo+d])*scale)),float(k[off-lane+d]),score[u]);
                    } else for(int d=0;d<4;++d)score[u]=fmaf(qd[d],float(k[off+d*32]),score[u]);
                } else score[u]=-INFINITY;
            }
            if(group!=7)for(int delta=16;delta;delta/=2)for(int u=0;u<2;++u)score[u]+=__shfl_down_sync(0xffffffff,score[u],delta,32);
            if(lane==0) {
                #pragma unroll
                for(int u=0;u<2;++u) if(base+u<bs) scores[warp][base+u]=score[u];
            }
        }
        }
        if(group==16)__syncthreads();else __syncwarp();
        float localmax=-INFINITY;
        for(int j=lane;j<bs;j+=32) localmax=fmaxf(localmax,scores[warp][j]);
        for(int delta=16;delta;delta/=2) localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
        float nm=fmaxf(maximum,localmax),alpha=expf(maximum-nm),blocksum=0;
        for(int j=lane;j<bs;j+=32) {
            float probability=active?expf(scores[warp][j]-nm):0.f;
            blocksum+=probability;scores[warp][j]=float(T(probability));
        }
        for(int delta=16;delta;delta/=2) blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
        denom=denom*alpha+blocksum;maximum=nm;
        for(int d=0;d<4;++d) acc[d]*=alpha;
        __syncwarp();
        if(active) for(int j=0;j<valid;++j) {
            float probability=scores[warp][j];
            int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane;
            for(int d=0;d<4;++d) acc[d]=fmaf(probability,float(v[off+d*32]),acc[d]);
        }
        __syncwarp();
    }
    if(active) for(int d=0;d<4;++d) out[qo+lane+d*32]=T(acc[d]/denom);
    return;
}
template<class T> __global__ __launch_bounds__(256)
void nsaPrefillSelectedPV4Cuda(const T* q,const T* k,const T* v,const int32_t* ids,
                              T* out,int tokens,int hq,int hkv,int width,int bs,float scale) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int group=7;int gh=blockIdx.z*8+warp,head=kv*group+gh;
    bool active=gh<group;
    int64_t qo=((int64_t(batch)*tokens+query)*hq+head)*128;
    int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ float scores[8][128];
    __shared__ float rounded_q[128][9];
    float qd[4]={0,0,0,0},acc[4]={0,0,0,0};
    if(active) for(int d=0;d<4;++d) qd[d]=float(T(float(q[qo+lane+d*32])*scale));
    // G7 preserves the author's dimension-ordered FMA, but assigns
    // independent token dots to lanes. Only Q and scores use shared memory.
    if((group==7 || group==16) && active)
        for(int d=lane;d<128;d+=32)
            rounded_q[d][warp]=float(T(float(q[qo+d])*scale));
    if(group==16)__syncthreads();else __syncwarp();
    float maximum=-INFINITY,denom=0;
    for(int slot=0;slot<width;++slot) {
        int id=ids[io+slot];
        if(id<0||int64_t(id)*bs>query) continue;
        int start=id*bs;
        int valid=min(bs,query+1-start);
        if(group==16)__syncthreads();
        if(group==7) {
            for(int j=lane;j<bs;j+=32) {
                float dot=-INFINITY;
                if(active && j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    dot=0.f;
                    for(int d=0;d<128;++d)
                        dot=fmaf(rounded_q[d][warp],float(k[kb+d]),dot);
                }
                scores[warp][j]=dot;
            }
        } else if(group==16) {
            for(int task=threadIdx.x;task<8*bs;task+=blockDim.x) {
                const int lh=task%8,j=task/8;
                float dot=-INFINITY;
                if(j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    float parts[32]={};
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        #pragma unroll
                        for(int l=0;l<32;++l)
                            parts[l]=fmaf(rounded_q[l+i*32][lh],float(k[kb+l+i*32]),parts[l]);
                    }
                    #pragma unroll
                    for(int delta=16;delta;delta/=2) {
                        #pragma unroll
                        for(int l=0;l<delta;++l)parts[l]+=parts[l+delta];
                    }
                    dot=parts[0];
                }
                scores[lh][j]=dot;
            }
        } else {
        // 2 independent token dots in flight; each retains its original
        // dimension FMA and shuffle reduction order. No additional K/V LDS.
        for(int base=0;base<bs;base+=2) {
            float score[2]={0};
            #pragma unroll
            for(int u=0;u<2;++u) {
                int j=base+u;
                if(active&&j<valid) {
                    int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane;
                    if(group==7) {
                      if(lane==0)for(int d=0;d<128;++d)score[u]=fmaf(float(T(float(q[qo+d])*scale)),float(k[off-lane+d]),score[u]);
                    } else for(int d=0;d<4;++d)score[u]=fmaf(qd[d],float(k[off+d*32]),score[u]);
                } else score[u]=-INFINITY;
            }
            if(group!=7)for(int delta=16;delta;delta/=2)for(int u=0;u<2;++u)score[u]+=__shfl_down_sync(0xffffffff,score[u],delta,32);
            if(lane==0) {
                #pragma unroll
                for(int u=0;u<2;++u) if(base+u<bs) scores[warp][base+u]=score[u];
            }
        }
        }
        if(group==16)__syncthreads();else __syncwarp();
        float localmax=-INFINITY;
        for(int j=lane;j<bs;j+=32) localmax=fmaxf(localmax,scores[warp][j]);
        for(int delta=16;delta;delta/=2) localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
        float nm=fmaxf(maximum,localmax),alpha=expf(maximum-nm),blocksum=0;
        for(int j=lane;j<bs;j+=32) {
            float probability=active?expf(scores[warp][j]-nm):0.f;
            blocksum+=probability;scores[warp][j]=float(T(probability));
        }
        for(int delta=16;delta;delta/=2) blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
        denom=denom*alpha+blocksum;maximum=nm;
        for(int d=0;d<4;++d) acc[d]*=alpha;
        __syncwarp();
        if(active) for(int j=0;j<valid;++j) {
            float probability=scores[warp][j];
            int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane*4;
            const uint2 packed=*reinterpret_cast<const uint2*>(v+off);
            T values[4];__builtin_memcpy(values,&packed,8);
            #pragma unroll
            for(int d=0;d<4;++d)acc[d]=fmaf(probability,float(values[d]),acc[d]);
        }
        __syncwarp();
    }
    if(active) for(int d=0;d<4;++d) out[qo+lane*4+d]=T(acc[d]/denom);
    return;
}
template<class T> __global__ __launch_bounds__(256)
void nsaPrefillSelectedG16PV4Cuda(const T* q,const T* k,const T* v,const int32_t* ids,
                              T* out,int tokens,int hq,int hkv,int width,int bs,float scale) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int group=16;int gh=blockIdx.z*8+warp,head=kv*group+gh;
    bool active=gh<group;
    int64_t qo=((int64_t(batch)*tokens+query)*hq+head)*128;
    int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ float scores[8][128];
    __shared__ float rounded_q[128][9];
    float qd[4]={0,0,0,0},acc[4]={0,0,0,0};
    if(active) for(int d=0;d<4;++d) qd[d]=float(T(float(q[qo+lane+d*32])*scale));
    // G7 preserves the author's dimension-ordered FMA, but assigns
    // independent token dots to lanes. Only Q and scores use shared memory.
    if((group==7 || group==16) && active)
        for(int d=lane;d<128;d+=32)
            rounded_q[d][warp]=float(T(float(q[qo+d])*scale));
    if(group==16)__syncthreads();else __syncwarp();
    float maximum=-INFINITY,denom=0;
    for(int slot=0;slot<width;++slot) {
        int id=ids[io+slot];
        if(id<0||int64_t(id)*bs>query) continue;
        int start=id*bs;
        int valid=min(bs,query+1-start);
        if(group==16)__syncthreads();
        if(group==7) {
            for(int j=lane;j<bs;j+=32) {
                float dot=-INFINITY;
                if(active && j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    dot=0.f;
                    for(int d=0;d<128;++d)
                        dot=fmaf(rounded_q[d][warp],float(k[kb+d]),dot);
                }
                scores[warp][j]=dot;
            }
        } else if(group==16) {
            for(int task=threadIdx.x;task<8*bs;task+=blockDim.x) {
                const int lh=task%8,j=task/8;
                float dot=-INFINITY;
                if(j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    float parts[32]={};
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        #pragma unroll
                        for(int l=0;l<32;++l)
                            parts[l]=fmaf(rounded_q[l+i*32][lh],float(k[kb+l+i*32]),parts[l]);
                    }
                    #pragma unroll
                    for(int delta=16;delta;delta/=2) {
                        #pragma unroll
                        for(int l=0;l<delta;++l)parts[l]+=parts[l+delta];
                    }
                    dot=parts[0];
                }
                scores[lh][j]=dot;
            }
        } else {
        // 2 independent token dots in flight; each retains its original
        // dimension FMA and shuffle reduction order. No additional K/V LDS.
        for(int base=0;base<bs;base+=2) {
            float score[2]={0};
            #pragma unroll
            for(int u=0;u<2;++u) {
                int j=base+u;
                if(active&&j<valid) {
                    int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane;
                    if(group==7) {
                      if(lane==0)for(int d=0;d<128;++d)score[u]=fmaf(float(T(float(q[qo+d])*scale)),float(k[off-lane+d]),score[u]);
                    } else for(int d=0;d<4;++d)score[u]=fmaf(qd[d],float(k[off+d*32]),score[u]);
                } else score[u]=-INFINITY;
            }
            if(group!=7)for(int delta=16;delta;delta/=2)for(int u=0;u<2;++u)score[u]+=__shfl_down_sync(0xffffffff,score[u],delta,32);
            if(lane==0) {
                #pragma unroll
                for(int u=0;u<2;++u) if(base+u<bs) scores[warp][base+u]=score[u];
            }
        }
        }
        if(group==16)__syncthreads();else __syncwarp();
        float localmax=-INFINITY;
        for(int j=lane;j<bs;j+=32) localmax=fmaxf(localmax,scores[warp][j]);
        for(int delta=16;delta;delta/=2) localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
        float nm=fmaxf(maximum,localmax),alpha=expf(maximum-nm),blocksum=0;
        for(int j=lane;j<bs;j+=32) {
            float probability=active?expf(scores[warp][j]-nm):0.f;
            blocksum+=probability;scores[warp][j]=float(T(probability));
        }
        for(int delta=16;delta;delta/=2) blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
        denom=denom*alpha+blocksum;maximum=nm;
        for(int d=0;d<4;++d) acc[d]*=alpha;
        __syncwarp();
        if(active) for(int j=0;j<valid;++j) {
            float probability=scores[warp][j];
            int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane*4;
            const uint2 packed=*reinterpret_cast<const uint2*>(v+off);
            T values[4];__builtin_memcpy(values,&packed,8);
            #pragma unroll
            for(int d=0;d<4;++d)acc[d]=fmaf(probability,float(values[d]),acc[d]);
        }
        __syncwarp();
    }
    if(active) for(int d=0;d<4;++d) out[qo+lane*4+d]=T(acc[d]/denom);
    return;
}
template<class T> __global__ __launch_bounds__(256)
void nsaPrefillSelectedG7BS64Cuda(const T* q,const T* k,const T* v,const int32_t* ids,
                              T* out,int tokens,int hq,int hkv,int width,int bs,float scale) {
    bs=64;
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int group=7;int gh=blockIdx.z*8+warp,head=kv*group+gh;
    bool active=gh<group;
    const int pv_head=threadIdx.x/16,pv_lane=threadIdx.x%16;
    const bool pv_active=pv_head<7;
    __shared__ float head_alpha[8],head_denom[8];
    int64_t qo=((int64_t(batch)*tokens+query)*hq+head)*128;
    int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ float scores[8][64];
    __shared__ float rounded_q[128][9];
    float qd[4]={0,0,0,0},acc[8]={};
    if(active) for(int d=0;d<4;++d) qd[d]=float(T(float(q[qo+lane+d*32])*scale));
    // G7 preserves the author's dimension-ordered FMA, but assigns
    // independent token dots to lanes. Only Q and scores use shared memory.
    if((group==7 || group==16) && active)
        for(int d=lane;d<128;d+=32)
            rounded_q[d][warp]=float(T(float(q[qo+d])*scale));
    if(lane==0)head_denom[warp]=0.f;
    __syncthreads();
    float maximum=-INFINITY,denom=0;
    for(int slot=0;slot<width;++slot) {
        int id=ids[io+slot];
        if(id<0||int64_t(id)*bs>query) continue;
        int start=id*bs;
        int valid=min(bs,query+1-start);
        if(group==16)__syncthreads();
        if(group==7) {
            for(int j=lane;j<bs;j+=32) {
                float dot=-INFINITY;
                if(active && j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    dot=0.f;
                    for(int d=0;d<128;++d)
                        dot=fmaf(rounded_q[d][warp],float(k[kb+d]),dot);
                }
                scores[warp][j]=dot;
            }
        } else if(group==16) {
            for(int task=threadIdx.x;task<8*bs;task+=blockDim.x) {
                const int lh=task%8,j=task/8;
                float dot=-INFINITY;
                if(j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    float parts[32]={};
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        #pragma unroll
                        for(int l=0;l<32;++l)
                            parts[l]=fmaf(rounded_q[l+i*32][lh],float(k[kb+l+i*32]),parts[l]);
                    }
                    #pragma unroll
                    for(int delta=16;delta;delta/=2) {
                        #pragma unroll
                        for(int l=0;l<delta;++l)parts[l]+=parts[l+delta];
                    }
                    dot=parts[0];
                }
                scores[lh][j]=dot;
            }
        } else {
        // 2 independent token dots in flight; each retains its original
        // dimension FMA and shuffle reduction order. No additional K/V LDS.
        for(int base=0;base<bs;base+=2) {
            float score[2]={0};
            #pragma unroll
            for(int u=0;u<2;++u) {
                int j=base+u;
                if(active&&j<valid) {
                    int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane;
                    if(group==7) {
                      if(lane==0)for(int d=0;d<128;++d)score[u]=fmaf(float(T(float(q[qo+d])*scale)),float(k[off-lane+d]),score[u]);
                    } else for(int d=0;d<4;++d)score[u]=fmaf(qd[d],float(k[off+d*32]),score[u]);
                } else score[u]=-INFINITY;
            }
            if(group!=7)for(int delta=16;delta;delta/=2)for(int u=0;u<2;++u)score[u]+=__shfl_down_sync(0xffffffff,score[u],delta,32);
            if(lane==0) {
                #pragma unroll
                for(int u=0;u<2;++u) if(base+u<bs) scores[warp][base+u]=score[u];
            }
        }
        }
        if(group==16)__syncthreads();else __syncwarp();
        float localmax=-INFINITY;
        for(int j=lane;j<bs;j+=32) localmax=fmaxf(localmax,scores[warp][j]);
        for(int delta=16;delta;delta/=2) localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
        float nm=fmaxf(maximum,localmax),alpha=expf(maximum-nm),blocksum=0;
        for(int j=lane;j<bs;j+=32) {
            float probability=active?expf(scores[warp][j]-nm):0.f;
            blocksum+=probability;scores[warp][j]=float(T(probability));
        }
        for(int delta=16;delta;delta/=2) blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
        denom=denom*alpha+blocksum;maximum=nm;
        if(lane==0) {head_alpha[warp]=alpha;head_denom[warp]=denom;}
        __syncthreads();
        if(pv_active) {
            #pragma unroll
            for(int d=0;d<8;++d)acc[d]*=head_alpha[pv_head];
            for(int j=0;j<valid;++j) {
                const float probability=scores[pv_head][j];
                const int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+pv_lane*8;
                const uint4 packed=*reinterpret_cast<const uint4*>(v+off);
                T values[8];__builtin_memcpy(values,&packed,16);
                #pragma unroll
                for(int d=0;d<8;++d)acc[d]=fmaf(probability,float(values[d]),acc[d]);
            }
        }
        __syncthreads();
    }
    if(pv_active) {
        const int64_t po=((int64_t(batch)*tokens+query)*hq+kv*7+pv_head)*128+pv_lane*8;
        #pragma unroll
        for(int d=0;d<8;++d)out[po+d]=T(acc[d]/head_denom[pv_head]);
    }
    return;
}
template<class T> __global__ __launch_bounds__(256)
void nsaPrefillSelectedG16VReuse2Cuda(const T* q,const T* k,const T* v,const int32_t* ids,
                              T* out,int tokens,int hq,int hkv,int width,int bs,float scale) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int group=16;int gh=blockIdx.z*8+warp,head=kv*group+gh;
    bool active=gh<group;
    const int pv_head=(threadIdx.x/32)*2;
    const bool pv_active=pv_head<8;
    int64_t qo=((int64_t(batch)*tokens+query)*hq+head)*128;
    int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ float scores[8][128];
    __shared__ float rounded_q[128][9];
    __shared__ float head_alpha[8],head_denom[8];
    float qd[4]={0,0,0,0},acc[8]={};
    if(active) for(int d=0;d<4;++d) qd[d]=float(T(float(q[qo+lane+d*32])*scale));
    // G7 preserves the author's dimension-ordered FMA, but assigns
    // independent token dots to lanes. Only Q and scores use shared memory.
    if((group==7 || group==16) && active)
        for(int d=lane;d<128;d+=32)
            rounded_q[d][warp]=float(T(float(q[qo+d])*scale));
    if(lane==0)head_denom[warp]=0.f;
    if(group==16)__syncthreads();else __syncwarp();
    float maximum=-INFINITY,denom=0;
    for(int slot=0;slot<width;++slot) {
        int id=ids[io+slot];
        if(id<0||int64_t(id)*bs>query) continue;
        int start=id*bs;
        int valid=min(bs,query+1-start);
        if(group==16)__syncthreads();
        if(group==7) {
            for(int j=lane;j<bs;j+=32) {
                float dot=-INFINITY;
                if(active && j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    dot=0.f;
                    for(int d=0;d<128;++d)
                        dot=fmaf(rounded_q[d][warp],float(k[kb+d]),dot);
                }
                scores[warp][j]=dot;
            }
        } else if(group==16) {
            for(int task=threadIdx.x;task<8*bs;task+=blockDim.x) {
                const int lh=task%8,j=task/8;
                float dot=-INFINITY;
                if(j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    float parts[32]={};
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        #pragma unroll
                        for(int l=0;l<32;++l)
                            parts[l]=fmaf(rounded_q[l+i*32][lh],float(k[kb+l+i*32]),parts[l]);
                    }
                    #pragma unroll
                    for(int delta=16;delta;delta/=2) {
                        #pragma unroll
                        for(int l=0;l<delta;++l)parts[l]+=parts[l+delta];
                    }
                    dot=parts[0];
                }
                scores[lh][j]=dot;
            }
        } else {
        // 2 independent token dots in flight; each retains its original
        // dimension FMA and shuffle reduction order. No additional K/V LDS.
        for(int base=0;base<bs;base+=2) {
            float score[2]={0};
            #pragma unroll
            for(int u=0;u<2;++u) {
                int j=base+u;
                if(active&&j<valid) {
                    int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane;
                    if(group==7) {
                      if(lane==0)for(int d=0;d<128;++d)score[u]=fmaf(float(T(float(q[qo+d])*scale)),float(k[off-lane+d]),score[u]);
                    } else for(int d=0;d<4;++d)score[u]=fmaf(qd[d],float(k[off+d*32]),score[u]);
                } else score[u]=-INFINITY;
            }
            if(group!=7)for(int delta=16;delta;delta/=2)for(int u=0;u<2;++u)score[u]+=__shfl_down_sync(0xffffffff,score[u],delta,32);
            if(lane==0) {
                #pragma unroll
                for(int u=0;u<2;++u) if(base+u<bs) scores[warp][base+u]=score[u];
            }
        }
        }
        if(group==16)__syncthreads();else __syncwarp();
        float localmax=-INFINITY;
        for(int j=lane;j<bs;j+=32) localmax=fmaxf(localmax,scores[warp][j]);
        for(int delta=16;delta;delta/=2) localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
        float nm=fmaxf(maximum,localmax),alpha=expf(maximum-nm),blocksum=0;
        for(int j=lane;j<bs;j+=32) {
            float probability=active?expf(scores[warp][j]-nm):0.f;
            blocksum+=probability;scores[warp][j]=float(T(probability));
        }
        for(int delta=16;delta;delta/=2) blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
        denom=denom*alpha+blocksum;maximum=nm;
        if(lane==0){head_alpha[warp]=alpha;head_denom[warp]=denom;}
        __syncthreads();
        if(pv_active) {
            #pragma unroll
            for(int h=0;h<2;++h)for(int d=0;d<4;++d)acc[h*4+d]*=head_alpha[pv_head+h];
            for(int j=0;j<valid;++j) {
                const int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane*4;
                const uint2 packed=*reinterpret_cast<const uint2*>(v+off);
                T values[4];__builtin_memcpy(values,&packed,8);
                #pragma unroll
                for(int h=0;h<2;++h) {
                    const float probability=scores[pv_head+h][j];
                    #pragma unroll
                    for(int d=0;d<4;++d)acc[h*4+d]=fmaf(probability,float(values[d]),acc[h*4+d]);
                }
            }
        }
        __syncthreads();
    }
    if(pv_active) {
        #pragma unroll
        for(int h=0;h<2;++h) {
            const int64_t po=((int64_t(batch)*tokens+query)*hq+kv*16+blockIdx.z*8+pv_head+h)*128+lane*4;
            #pragma unroll
            for(int d=0;d<4;++d)out[po+d]=T(acc[h*4+d]/head_denom[pv_head+h]);
        }
    }
    return;
}
template<class T> __global__ __launch_bounds__(256)
void nsaPrefillSelectedG16PvUnroll8Cuda(const T* q,const T* k,const T* v,const int32_t* ids,
                              T* out,int tokens,int hq,int hkv,int width,int bs,float scale) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int group=16;int gh=blockIdx.z*8+warp,head=kv*group+gh;
    bool active=gh<group;
    const int pv_head=(threadIdx.x/32)*2;
    const bool pv_active=pv_head<8;
    int64_t qo=((int64_t(batch)*tokens+query)*hq+head)*128;
    int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ float scores[8][128];
    __shared__ float rounded_q[128][9];
    __shared__ float head_alpha[8],head_denom[8];
    float qd[4]={0,0,0,0},acc[8]={};
    if(active) for(int d=0;d<4;++d) qd[d]=float(T(float(q[qo+lane+d*32])*scale));
    // G7 preserves the author's dimension-ordered FMA, but assigns
    // independent token dots to lanes. Only Q and scores use shared memory.
    if((group==7 || group==16) && active)
        for(int d=lane;d<128;d+=32)
            rounded_q[d][warp]=float(T(float(q[qo+d])*scale));
    if(lane==0)head_denom[warp]=0.f;
    if(group==16)__syncthreads();else __syncwarp();
    float maximum=-INFINITY,denom=0;
    for(int slot=0;slot<width;++slot) {
        int id=ids[io+slot];
        if(id<0||int64_t(id)*bs>query) continue;
        int start=id*bs;
        int valid=min(bs,query+1-start);
        if(group==16)__syncthreads();
        if(group==7) {
            for(int j=lane;j<bs;j+=32) {
                float dot=-INFINITY;
                if(active && j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    dot=0.f;
                    for(int d=0;d<128;++d)
                        dot=fmaf(rounded_q[d][warp],float(k[kb+d]),dot);
                }
                scores[warp][j]=dot;
            }
        } else if(group==16) {
            for(int task=threadIdx.x;task<8*bs;task+=blockDim.x) {
                const int lh=task%8,j=task/8;
                float dot=-INFINITY;
                if(j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    float parts[32]={};
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        #pragma unroll
                        for(int l=0;l<32;++l)
                            parts[l]=fmaf(rounded_q[l+i*32][lh],float(k[kb+l+i*32]),parts[l]);
                    }
                    #pragma unroll
                    for(int delta=16;delta;delta/=2) {
                        #pragma unroll
                        for(int l=0;l<delta;++l)parts[l]+=parts[l+delta];
                    }
                    dot=parts[0];
                }
                scores[lh][j]=dot;
            }
        } else {
        // 2 independent token dots in flight; each retains its original
        // dimension FMA and shuffle reduction order. No additional K/V LDS.
        for(int base=0;base<bs;base+=2) {
            float score[2]={0};
            #pragma unroll
            for(int u=0;u<2;++u) {
                int j=base+u;
                if(active&&j<valid) {
                    int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane;
                    if(group==7) {
                      if(lane==0)for(int d=0;d<128;++d)score[u]=fmaf(float(T(float(q[qo+d])*scale)),float(k[off-lane+d]),score[u]);
                    } else for(int d=0;d<4;++d)score[u]=fmaf(qd[d],float(k[off+d*32]),score[u]);
                } else score[u]=-INFINITY;
            }
            if(group!=7)for(int delta=16;delta;delta/=2)for(int u=0;u<2;++u)score[u]+=__shfl_down_sync(0xffffffff,score[u],delta,32);
            if(lane==0) {
                #pragma unroll
                for(int u=0;u<2;++u) if(base+u<bs) scores[warp][base+u]=score[u];
            }
        }
        }
        if(group==16)__syncthreads();else __syncwarp();
        float localmax=-INFINITY;
        for(int j=lane;j<bs;j+=32) localmax=fmaxf(localmax,scores[warp][j]);
        for(int delta=16;delta;delta/=2) localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
        float nm=fmaxf(maximum,localmax),alpha=expf(maximum-nm),blocksum=0;
        for(int j=lane;j<bs;j+=32) {
            float probability=active?expf(scores[warp][j]-nm):0.f;
            blocksum+=probability;scores[warp][j]=float(T(probability));
        }
        for(int delta=16;delta;delta/=2) blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
        denom=denom*alpha+blocksum;maximum=nm;
        if(lane==0){head_alpha[warp]=alpha;head_denom[warp]=denom;}
        __syncthreads();
        if(pv_active) {
            #pragma unroll
            for(int h=0;h<2;++h)for(int d=0;d<4;++d)acc[h*4+d]*=head_alpha[pv_head+h];
            #pragma unroll 8
            for(int j=0;j<valid;++j) {
                const int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane*4;
                const uint2 packed=*reinterpret_cast<const uint2*>(v+off);
                T values[4];__builtin_memcpy(values,&packed,8);
                #pragma unroll
                for(int h=0;h<2;++h) {
                    const float probability=scores[pv_head+h][j];
                    #pragma unroll
                    for(int d=0;d<4;++d)acc[h*4+d]=fmaf(probability,float(values[d]),acc[h*4+d]);
                }
            }
        }
        __syncthreads();
    }
    if(pv_active) {
        #pragma unroll
        for(int h=0;h<2;++h) {
            const int64_t po=((int64_t(batch)*tokens+query)*hq+kv*16+blockIdx.z*8+pv_head+h)*128+lane*4;
            #pragma unroll
            for(int d=0;d<4;++d)out[po+d]=T(acc[h*4+d]/head_denom[pv_head+h]);
        }
    }
    return;
}
template<class T> __global__ __launch_bounds__(256)
void nsaPrefillSelectedG16VReuse4D2Cuda(const T* q,const T* k,const T* v,const int32_t* ids,
                              T* out,int tokens,int hq,int hkv,int width,int bs,float scale) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int group=16;int gh=blockIdx.z*8+warp,head=kv*group+gh;
    bool active=gh<group;
    const int pv_lane=threadIdx.x%64;
    const int pv_head=(threadIdx.x/64)*4;
    const bool pv_active=pv_head<8;
    int64_t qo=((int64_t(batch)*tokens+query)*hq+head)*128;
    int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ float scores[8][128];
    __shared__ float rounded_q[128][9];
    __shared__ float head_alpha[8],head_denom[8];
    float qd[4]={0,0,0,0},acc[8]={};
    if(active) for(int d=0;d<4;++d) qd[d]=float(T(float(q[qo+lane+d*32])*scale));
    // G7 preserves the author's dimension-ordered FMA, but assigns
    // independent token dots to lanes. Only Q and scores use shared memory.
    if((group==7 || group==16) && active)
        for(int d=lane;d<128;d+=32)
            rounded_q[d][warp]=float(T(float(q[qo+d])*scale));
    if(lane==0)head_denom[warp]=0.f;
    if(group==16)__syncthreads();else __syncwarp();
    float maximum=-INFINITY,denom=0;
    for(int slot=0;slot<width;++slot) {
        int id=ids[io+slot];
        if(id<0||int64_t(id)*bs>query) continue;
        int start=id*bs;
        int valid=min(bs,query+1-start);
        if(group==16)__syncthreads();
        if(group==7) {
            for(int j=lane;j<bs;j+=32) {
                float dot=-INFINITY;
                if(active && j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    dot=0.f;
                    for(int d=0;d<128;++d)
                        dot=fmaf(rounded_q[d][warp],float(k[kb+d]),dot);
                }
                scores[warp][j]=dot;
            }
        } else if(group==16) {
            for(int task=threadIdx.x;task<8*bs;task+=blockDim.x) {
                const int lh=task%8,j=task/8;
                float dot=-INFINITY;
                if(j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    float parts[32]={};
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        #pragma unroll
                        for(int l=0;l<32;++l)
                            parts[l]=fmaf(rounded_q[l+i*32][lh],float(k[kb+l+i*32]),parts[l]);
                    }
                    #pragma unroll
                    for(int delta=16;delta;delta/=2) {
                        #pragma unroll
                        for(int l=0;l<delta;++l)parts[l]+=parts[l+delta];
                    }
                    dot=parts[0];
                }
                scores[lh][j]=dot;
            }
        } else {
        // 2 independent token dots in flight; each retains its original
        // dimension FMA and shuffle reduction order. No additional K/V LDS.
        for(int base=0;base<bs;base+=2) {
            float score[2]={0};
            #pragma unroll
            for(int u=0;u<2;++u) {
                int j=base+u;
                if(active&&j<valid) {
                    int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane;
                    if(group==7) {
                      if(lane==0)for(int d=0;d<128;++d)score[u]=fmaf(float(T(float(q[qo+d])*scale)),float(k[off-lane+d]),score[u]);
                    } else for(int d=0;d<4;++d)score[u]=fmaf(qd[d],float(k[off+d*32]),score[u]);
                } else score[u]=-INFINITY;
            }
            if(group!=7)for(int delta=16;delta;delta/=2)for(int u=0;u<2;++u)score[u]+=__shfl_down_sync(0xffffffff,score[u],delta,32);
            if(lane==0) {
                #pragma unroll
                for(int u=0;u<2;++u) if(base+u<bs) scores[warp][base+u]=score[u];
            }
        }
        }
        if(group==16)__syncthreads();else __syncwarp();
        float localmax=-INFINITY;
        for(int j=lane;j<bs;j+=32) localmax=fmaxf(localmax,scores[warp][j]);
        for(int delta=16;delta;delta/=2) localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
        float nm=fmaxf(maximum,localmax),alpha=expf(maximum-nm),blocksum=0;
        for(int j=lane;j<bs;j+=32) {
            float probability=active?expf(scores[warp][j]-nm):0.f;
            blocksum+=probability;scores[warp][j]=float(T(probability));
        }
        for(int delta=16;delta;delta/=2) blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
        denom=denom*alpha+blocksum;maximum=nm;
        if(lane==0){head_alpha[warp]=alpha;head_denom[warp]=denom;}
        __syncthreads();
        if(pv_active) {
            #pragma unroll
            for(int h=0;h<4;++h)for(int d=0;d<2;++d)acc[h*2+d]*=head_alpha[pv_head+h];
            for(int j=0;j<valid;++j) {
                const int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+pv_lane*2;
                const uint32_t packed=*reinterpret_cast<const uint32_t*>(v+off);
                T values[2];__builtin_memcpy(values,&packed,4);
                #pragma unroll
                for(int h=0;h<4;++h) {
                    const float probability=scores[pv_head+h][j];
                    #pragma unroll
                    for(int d=0;d<2;++d)acc[h*2+d]=fmaf(probability,float(values[d]),acc[h*2+d]);
                }
            }
        }
        __syncthreads();
    }
    if(pv_active) {
        #pragma unroll
        for(int h=0;h<4;++h) {
            const int64_t po=((int64_t(batch)*tokens+query)*hq+kv*16+blockIdx.z*8+pv_head+h)*128+pv_lane*2;
            #pragma unroll
            for(int d=0;d<2;++d)out[po+d]=T(acc[h*2+d]/head_denom[pv_head+h]);
        }
    }
    return;
}

template<class T> __global__ __launch_bounds__(512)
void nsaPrefillSelectedG16H16PV4Cuda(const T* q,const T* k,const T* v,const int32_t* ids,
                              T* out,int tokens,int hq,int hkv,int width,int bs,float scale) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int group=16;int gh=blockIdx.z*16+warp,head=kv*group+gh;
    bool active=gh<group;
    const int pv_lane=threadIdx.x%64;
    const int pv_head=(threadIdx.x/64)*4;
    const bool pv_active=pv_head<16;
    int64_t qo=((int64_t(batch)*tokens+query)*hq+head)*128;
    int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ float scores[16][64];
    __shared__ float rounded_q[128][17];
    __shared__ float head_alpha[16],head_denom[16];
    float qd[4]={0,0,0,0},acc[8]={};
    if(active) for(int d=0;d<4;++d) qd[d]=float(T(float(q[qo+lane+d*32])*scale));
    // G7 preserves the author's dimension-ordered FMA, but assigns
    // independent token dots to lanes. Only Q and scores use shared memory.
    if((group==7 || group==16) && active)
        for(int d=lane;d<128;d+=32)
            rounded_q[d][warp]=float(T(float(q[qo+d])*scale));
    if(lane==0)head_denom[warp]=0.f;
    if(group==16)__syncthreads();else __syncwarp();
    float maximum=-INFINITY,denom=0;
    for(int slot=0;slot<width;++slot) {
        int id=ids[io+slot];
        if(id<0||int64_t(id)*bs>query) continue;
        int start=id*bs;
        int valid=min(bs,query+1-start);
        if(group==16)__syncthreads();
        if(group==7) {
            for(int j=lane;j<bs;j+=32) {
                float dot=-INFINITY;
                if(active && j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    dot=0.f;
                    for(int d=0;d<128;++d)
                        dot=fmaf(rounded_q[d][warp],float(k[kb+d]),dot);
                }
                scores[warp][j]=dot;
            }
        } else if(group==16) {
            for(int task=threadIdx.x;task<16*bs;task+=blockDim.x) {
                const int lh=task%16,j=task/16;
                float dot=-INFINITY;
                if(j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    float parts[32]={};
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        #pragma unroll
                        for(int l=0;l<32;++l)
                            parts[l]=fmaf(rounded_q[l+i*32][lh],float(k[kb+l+i*32]),parts[l]);
                    }
                    #pragma unroll
                    for(int delta=16;delta;delta/=2) {
                        #pragma unroll
                        for(int l=0;l<delta;++l)parts[l]+=parts[l+delta];
                    }
                    dot=parts[0];
                }
                scores[lh][j]=dot;
            }
        } else {
        // 2 independent token dots in flight; each retains its original
        // dimension FMA and shuffle reduction order. No additional K/V LDS.
        for(int base=0;base<bs;base+=2) {
            float score[2]={0};
            #pragma unroll
            for(int u=0;u<2;++u) {
                int j=base+u;
                if(active&&j<valid) {
                    int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane;
                    if(group==7) {
                      if(lane==0)for(int d=0;d<128;++d)score[u]=fmaf(float(T(float(q[qo+d])*scale)),float(k[off-lane+d]),score[u]);
                    } else for(int d=0;d<4;++d)score[u]=fmaf(qd[d],float(k[off+d*32]),score[u]);
                } else score[u]=-INFINITY;
            }
            if(group!=7)for(int delta=16;delta;delta/=2)for(int u=0;u<2;++u)score[u]+=__shfl_down_sync(0xffffffff,score[u],delta,32);
            if(lane==0) {
                #pragma unroll
                for(int u=0;u<2;++u) if(base+u<bs) scores[warp][base+u]=score[u];
            }
        }
        }
        if(group==16)__syncthreads();else __syncwarp();
        float localmax=-INFINITY;
        for(int j=lane;j<bs;j+=32) localmax=fmaxf(localmax,scores[warp][j]);
        for(int delta=16;delta;delta/=2) localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
        float nm=fmaxf(maximum,localmax),alpha=expf(maximum-nm),blocksum=0;
        for(int j=lane;j<bs;j+=32) {
            float probability=active?expf(scores[warp][j]-nm):0.f;
            blocksum+=probability;scores[warp][j]=float(T(probability));
        }
        for(int delta=16;delta;delta/=2) blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
        denom=denom*alpha+blocksum;maximum=nm;
        if(lane==0){head_alpha[warp]=alpha;head_denom[warp]=denom;}
        __syncthreads();
        if(pv_active) {
            #pragma unroll
            for(int h=0;h<4;++h)for(int d=0;d<2;++d)acc[h*2+d]*=head_alpha[pv_head+h];
            #pragma unroll 8
            for(int j=0;j<valid;++j) {
                const int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+pv_lane*2;
                const uint32_t packed=*reinterpret_cast<const uint32_t*>(v+off);
                T values[2];__builtin_memcpy(values,&packed,4);
                #pragma unroll
                for(int h=0;h<4;++h) {
                    const float probability=scores[pv_head+h][j];
                    #pragma unroll
                    for(int d=0;d<2;++d)acc[h*2+d]=fmaf(probability,float(values[d]),acc[h*2+d]);
                }
            }
        }
        __syncthreads();
    }
    if(pv_active) {
        #pragma unroll
        for(int h=0;h<4;++h) {
            const int64_t po=((int64_t(batch)*tokens+query)*hq+kv*16+blockIdx.z*16+pv_head+h)*128+pv_lane*2;
            #pragma unroll
            for(int d=0;d<2;++d)out[po+d]=T(acc[h*2+d]/head_denom[pv_head+h]);
        }
    }
    return;
}
extern "C" int personal_nsa_prefill_selected(const void* q,const void* k,const void* v,
 const int32_t* ids,void* out,int b,int t,int hq,int hkv,int width,int bs,float scale,int dtype,void* stream) {
    if(b<=0||t<=0||hq<=0||hkv<=0||hq%hkv||width<=0||bs<=0||bs>128||dtype<0||dtype>1) return -1;
    dim3 grid(b*t,hkv,(hq/hkv+7)/8);
    if(hq/hkv==7 && bs==64 && t>=1024 && (reinterpret_cast<uintptr_t>(v)&15)==0) {
      if(dtype==0)nsaPrefillSelectedG7BS64Cuda<<<grid,256,0,(cudaStream_t)stream>>>((const half*)q,(const half*)k,(const half*)v,ids,(half*)out,t,hq,hkv,width,bs,scale);
      else nsaPrefillSelectedG7BS64Cuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,ids,(__nv_bfloat16*)out,t,hq,hkv,width,bs,scale);
    } else if(hq/hkv==16 && bs==64 && t>=1024 && (reinterpret_cast<uintptr_t>(v)&7)==0) {
      if(dtype==0)nsaPrefillSelectedG16H16PV4Cuda<<<dim3(b*t,hkv,1),512,0,(cudaStream_t)stream>>>((const half*)q,(const half*)k,(const half*)v,ids,(half*)out,t,hq,hkv,width,bs,scale);
      else nsaPrefillSelectedG16VReuse4D2Cuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,ids,(__nv_bfloat16*)out,t,hq,hkv,width,bs,scale);
    } else     if(hq/hkv==7 && t>=1024 && (reinterpret_cast<uintptr_t>(v)&7)==0) {
    if(dtype==0) nsaPrefillSelectedPV4Cuda<<<grid,256,0,(cudaStream_t)stream>>>((const half*)q,(const half*)k,(const half*)v,ids,(half*)out,t,hq,hkv,width,bs,scale);
    else nsaPrefillSelectedPV4Cuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,ids,(__nv_bfloat16*)out,t,hq,hkv,width,bs,scale);
    } else {
    if(dtype==0) nsaPrefillSelectedWarpCuda<<<grid,256,0,(cudaStream_t)stream>>>((const half*)q,(const half*)k,(const half*)v,ids,(half*)out,t,hq,hkv,width,bs,scale);
    else nsaPrefillSelectedWarpCuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,ids,(__nv_bfloat16*)out,t,hq,hkv,width,bs,scale);
    }
    return int(cudaGetLastError());
}

template<class T> __global__ void nsaPackK64Cuda(const T* input,T* packed,int tokens,int hkv) {
 __shared__ T tile[32][33];
 int x=threadIdx.x,y=threadIdx.y,batch=blockIdx.z/hkv,kv=blockIdx.z%hkv;
 for(int j=0;j<32;j+=8) {
  int token=blockIdx.y*32+y+j,d=blockIdx.x*32+x;
  tile[y+j][x]=token<tokens?input[((int64_t(batch)*tokens+token)*hkv+kv)*128+d]:T(0.f);
 }
 __syncthreads();
 for(int j=0;j<32;j+=8) {
  int token=blockIdx.y*32+x,d=blockIdx.x*32+y+j;
  int64_t off=(((int64_t(batch)*hkv+kv)*((tokens+63)/64)+token/64)*128+d)*64+token%64;
  packed[off]=tile[x][y+j];
 }
}
template<class T> __global__ __launch_bounds__(256)
void nsaPrefillSelectedG7PackedCuda(const T* q,const T* k,const T* v,const int32_t* ids,
                              T* out,int tokens,int hq,int hkv,int width,int bs,float scale) {
    bs=64;
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int group=7;int gh=blockIdx.z*8+warp,head=kv*group+gh;
    bool active=gh<group;
    const int pv_head=threadIdx.x/16,pv_lane=threadIdx.x%16;
    const bool pv_active=pv_head<7;
    __shared__ float head_alpha[8],head_denom[8];
    int64_t qo=((int64_t(batch)*tokens+query)*hq+head)*128;
    int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ float scores[8][64];
    __shared__ float rounded_q[128][9];
    float qd[4]={0,0,0,0},acc[8]={};
    if(active) for(int d=0;d<4;++d) qd[d]=float(T(float(q[qo+lane+d*32])*scale));
    // G7 preserves the author's dimension-ordered FMA, but assigns
    // independent token dots to lanes. Only Q and scores use shared memory.
    if((group==7 || group==16) && active)
        for(int d=lane;d<128;d+=32)
            rounded_q[d][warp]=float(T(float(q[qo+d])*scale));
    if(lane==0)head_denom[warp]=0.f;
    __syncthreads();
    float maximum=-INFINITY,denom=0;
    for(int slot=0;slot<width;++slot) {
        int id=ids[io+slot];
        if(id<0||int64_t(id)*bs>query) continue;
        int start=id*bs;
        int valid=min(bs,query+1-start);
        if(group==16)__syncthreads();
        if(group==7) {
            for(int j=lane;j<bs;j+=32) {
                float dot=-INFINITY;
                if(active && j<valid) {
                    const int64_t kb=(((int64_t(batch)*hkv+kv)*((tokens+63)/64)+id)*128)*64+j;
                    dot=0.f;
                    for(int d=0;d<128;++d)
                        dot=fmaf(rounded_q[d][warp],float(k[kb+d*64]),dot);
                }
                scores[warp][j]=dot;
            }
        } else if(group==16) {
            for(int task=threadIdx.x;task<8*bs;task+=blockDim.x) {
                const int lh=task%8,j=task/8;
                float dot=-INFINITY;
                if(j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    float parts[32]={};
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        #pragma unroll
                        for(int l=0;l<32;++l)
                            parts[l]=fmaf(rounded_q[l+i*32][lh],float(k[kb+l+i*32]),parts[l]);
                    }
                    #pragma unroll
                    for(int delta=16;delta;delta/=2) {
                        #pragma unroll
                        for(int l=0;l<delta;++l)parts[l]+=parts[l+delta];
                    }
                    dot=parts[0];
                }
                scores[lh][j]=dot;
            }
        } else {
        // 2 independent token dots in flight; each retains its original
        // dimension FMA and shuffle reduction order. No additional K/V LDS.
        for(int base=0;base<bs;base+=2) {
            float score[2]={0};
            #pragma unroll
            for(int u=0;u<2;++u) {
                int j=base+u;
                if(active&&j<valid) {
                    int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane;
                    if(group==7) {
                      if(lane==0)for(int d=0;d<128;++d)score[u]=fmaf(float(T(float(q[qo+d])*scale)),float(k[off-lane+d]),score[u]);
                    } else for(int d=0;d<4;++d)score[u]=fmaf(qd[d],float(k[off+d*32]),score[u]);
                } else score[u]=-INFINITY;
            }
            if(group!=7)for(int delta=16;delta;delta/=2)for(int u=0;u<2;++u)score[u]+=__shfl_down_sync(0xffffffff,score[u],delta,32);
            if(lane==0) {
                #pragma unroll
                for(int u=0;u<2;++u) if(base+u<bs) scores[warp][base+u]=score[u];
            }
        }
        }
        if(group==16)__syncthreads();else __syncwarp();
        float localmax=-INFINITY;
        for(int j=lane;j<bs;j+=32) localmax=fmaxf(localmax,scores[warp][j]);
        for(int delta=16;delta;delta/=2) localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
        float nm=fmaxf(maximum,localmax),alpha=expf(maximum-nm),blocksum=0;
        for(int j=lane;j<bs;j+=32) {
            float probability=active?expf(scores[warp][j]-nm):0.f;
            blocksum+=probability;scores[warp][j]=float(T(probability));
        }
        for(int delta=16;delta;delta/=2) blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
        denom=denom*alpha+blocksum;maximum=nm;
        if(lane==0) {head_alpha[warp]=alpha;head_denom[warp]=denom;}
        __syncthreads();
        if(pv_active) {
            #pragma unroll
            for(int d=0;d<8;++d)acc[d]*=head_alpha[pv_head];
            for(int j=0;j<valid;++j) {
                const float probability=scores[pv_head][j];
                const int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+pv_lane*8;
                const uint4 packed=*reinterpret_cast<const uint4*>(v+off);
                T values[8];__builtin_memcpy(values,&packed,16);
                #pragma unroll
                for(int d=0;d<8;++d)acc[d]=fmaf(probability,float(values[d]),acc[d]);
            }
        }
        __syncthreads();
    }
    if(pv_active) {
        const int64_t po=((int64_t(batch)*tokens+query)*hq+kv*7+pv_head)*128+pv_lane*8;
        #pragma unroll
        for(int d=0;d<8;++d)out[po+d]=T(acc[d]/head_denom[pv_head]);
    }
    return;
}
extern "C" int personal_nsa_selected_packed(const void* q,const void* k,const void* v,void* packed,
 const int32_t* ids,void* out,int b,int t,int hq,int hkv,int width,int bs,float scale,int dtype,void* stream) {
 if(b<=0||t<=0||hq<=0||hkv<=0||hq%hkv||width<=0||bs<=0||bs>128||dtype<0||dtype>1)return -1;
 if(hq/hkv!=7 || bs!=64 || t<1024 || (reinterpret_cast<uintptr_t>(v)&15)!=0)
  return personal_nsa_prefill_selected(q,k,v,ids,out,b,t,hq,hkv,width,bs,scale,dtype,stream);
 if(!packed)return -2;
 dim3 pg(4,2*((t+63)/64),b*hkv),pt(32,8),grid(b*t,hkv,1);
 if(dtype==0) {
  nsaPackK64Cuda<<<pg,pt,0,(cudaStream_t)stream>>>((const half*)k,(half*)packed,t,hkv);
  nsaPrefillSelectedG7PackedCuda<<<grid,256,0,(cudaStream_t)stream>>>((const half*)q,(const half*)packed,(const half*)v,ids,(half*)out,t,hq,hkv,width,bs,scale);
 } else {
  nsaPackK64Cuda<<<pg,pt,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)k,(__nv_bfloat16*)packed,t,hkv);
  nsaPrefillSelectedG7PackedCuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)packed,(const __nv_bfloat16*)v,ids,(__nv_bfloat16*)out,t,hq,hkv,width,bs,scale);
 }
 return int(cudaGetLastError());
}

template<class T> __global__ __launch_bounds__(256)
void nsaSelectedBF16Unroll4Cuda(const T* q,const T* k,const T* v,const int32_t* ids,
                              T* out,int tokens,int hq,int hkv,int width,int bs,float scale) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int group=16;int gh=blockIdx.z*8+warp,head=kv*group+gh;
    bool active=gh<group;
    const int pv_lane=threadIdx.x%64;
    const int pv_head=(threadIdx.x/64)*4;
    const bool pv_active=pv_head<8;
    int64_t qo=((int64_t(batch)*tokens+query)*hq+head)*128;
    int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ float scores[8][128];
    __shared__ float rounded_q[128][9];
    __shared__ float head_alpha[8],head_denom[8];
    float qd[4]={0,0,0,0},acc[8]={};
    if(active) for(int d=0;d<4;++d) qd[d]=float(T(float(q[qo+lane+d*32])*scale));
    // G7 preserves the author's dimension-ordered FMA, but assigns
    // independent token dots to lanes. Only Q and scores use shared memory.
    if((group==7 || group==16) && active)
        for(int d=lane;d<128;d+=32)
            rounded_q[d][warp]=float(T(float(q[qo+d])*scale));
    if(lane==0)head_denom[warp]=0.f;
    if(group==16)__syncthreads();else __syncwarp();
    float maximum=-INFINITY,denom=0;
    for(int slot=0;slot<width;++slot) {
        int id=ids[io+slot];
        if(id<0||int64_t(id)*bs>query) continue;
        int start=id*bs;
        int valid=min(bs,query+1-start);
        if(group==16)__syncthreads();
        if(group==7) {
            for(int j=lane;j<bs;j+=32) {
                float dot=-INFINITY;
                if(active && j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    dot=0.f;
                    for(int d=0;d<128;++d)
                        dot=fmaf(rounded_q[d][warp],float(k[kb+d]),dot);
                }
                scores[warp][j]=dot;
            }
        } else if(group==16) {
            for(int task=threadIdx.x;task<8*bs;task+=blockDim.x) {
                const int lh=task%8,j=task/8;
                float dot=-INFINITY;
                if(j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    float parts[32]={};
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        #pragma unroll
                        for(int l=0;l<32;++l)
                            parts[l]=fmaf(rounded_q[l+i*32][lh],float(k[kb+l+i*32]),parts[l]);
                    }
                    #pragma unroll
                    for(int delta=16;delta;delta/=2) {
                        #pragma unroll
                        for(int l=0;l<delta;++l)parts[l]+=parts[l+delta];
                    }
                    dot=parts[0];
                }
                scores[lh][j]=dot;
            }
        } else {
        // 2 independent token dots in flight; each retains its original
        // dimension FMA and shuffle reduction order. No additional K/V LDS.
        for(int base=0;base<bs;base+=2) {
            float score[2]={0};
            #pragma unroll
            for(int u=0;u<2;++u) {
                int j=base+u;
                if(active&&j<valid) {
                    int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane;
                    if(group==7) {
                      if(lane==0)for(int d=0;d<128;++d)score[u]=fmaf(float(T(float(q[qo+d])*scale)),float(k[off-lane+d]),score[u]);
                    } else for(int d=0;d<4;++d)score[u]=fmaf(qd[d],float(k[off+d*32]),score[u]);
                } else score[u]=-INFINITY;
            }
            if(group!=7)for(int delta=16;delta;delta/=2)for(int u=0;u<2;++u)score[u]+=__shfl_down_sync(0xffffffff,score[u],delta,32);
            if(lane==0) {
                #pragma unroll
                for(int u=0;u<2;++u) if(base+u<bs) scores[warp][base+u]=score[u];
            }
        }
        }
        if(group==16)__syncthreads();else __syncwarp();
        float localmax=-INFINITY;
        for(int j=lane;j<bs;j+=32) localmax=fmaxf(localmax,scores[warp][j]);
        for(int delta=16;delta;delta/=2) localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
        float nm=fmaxf(maximum,localmax),alpha=expf(maximum-nm),blocksum=0;
        for(int j=lane;j<bs;j+=32) {
            float probability=active?expf(scores[warp][j]-nm):0.f;
            blocksum+=probability;scores[warp][j]=float(T(probability));
        }
        for(int delta=16;delta;delta/=2) blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
        denom=denom*alpha+blocksum;maximum=nm;
        if(lane==0){head_alpha[warp]=alpha;head_denom[warp]=denom;}
        __syncthreads();
        if(pv_active) {
            #pragma unroll
            for(int h=0;h<4;++h)for(int d=0;d<2;++d)acc[h*2+d]*=head_alpha[pv_head+h];
            #pragma unroll 4
            for(int j=0;j<valid;++j) {
                const int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+pv_lane*2;
                const uint32_t packed=*reinterpret_cast<const uint32_t*>(v+off);
                T values[2];__builtin_memcpy(values,&packed,4);
                #pragma unroll
                for(int h=0;h<4;++h) {
                    const float probability=scores[pv_head+h][j];
                    #pragma unroll
                    for(int d=0;d<2;++d)acc[h*2+d]=fmaf(probability,float(values[d]),acc[h*2+d]);
                }
            }
        }
        __syncthreads();
    }
    if(pv_active) {
        #pragma unroll
        for(int h=0;h<4;++h) {
            const int64_t po=((int64_t(batch)*tokens+query)*hq+kv*16+blockIdx.z*8+pv_head+h)*128+pv_lane*2;
            #pragma unroll
            for(int d=0;d<2;++d)out[po+d]=T(acc[h*2+d]/head_denom[pv_head+h]);
        }
    }
    return;
}

extern "C" int personal_nsa_selected_packed_r46_bf16u4(const void* q,const void* k,const void* v,void* scratch,
 const int32_t* ids,void* out,int b,int t,int hq,int hkv,int width,int bs,float scale,int dtype,void* stream) {
    if(dtype==1&&hq/hkv==16&&bs==64&&t>=1024&&(reinterpret_cast<uintptr_t>(v)&7)==0) {
        dim3 grid(b*t,hkv,2);
        nsaSelectedBF16Unroll4Cuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,
            (const __nv_bfloat16*)v,ids,(__nv_bfloat16*)out,t,hq,hkv,width,bs,scale);
        return int(cudaGetLastError());
    }
    return personal_nsa_selected_packed(q,k,v,scratch,ids,out,b,t,hq,hkv,width,bs,scale,dtype,stream);
}

template<class T> __global__ __launch_bounds__(256)
void nsaSelectedBF16Unroll8Cuda(const T* q,const T* k,const T* v,const int32_t* ids,
                              T* out,int tokens,int hq,int hkv,int width,int bs,float scale) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int group=16;int gh=blockIdx.z*8+warp,head=kv*group+gh;
    bool active=gh<group;
    const int pv_lane=threadIdx.x%64;
    const int pv_head=(threadIdx.x/64)*4;
    const bool pv_active=pv_head<8;
    int64_t qo=((int64_t(batch)*tokens+query)*hq+head)*128;
    int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ float scores[8][128];
    __shared__ float rounded_q[128][9];
    __shared__ float head_alpha[8],head_denom[8];
    float qd[4]={0,0,0,0},acc[8]={};
    if(active) for(int d=0;d<4;++d) qd[d]=float(T(float(q[qo+lane+d*32])*scale));
    // G7 preserves the author's dimension-ordered FMA, but assigns
    // independent token dots to lanes. Only Q and scores use shared memory.
    if((group==7 || group==16) && active)
        for(int d=lane;d<128;d+=32)
            rounded_q[d][warp]=float(T(float(q[qo+d])*scale));
    if(lane==0)head_denom[warp]=0.f;
    if(group==16)__syncthreads();else __syncwarp();
    float maximum=-INFINITY,denom=0;
    for(int slot=0;slot<width;++slot) {
        int id=ids[io+slot];
        if(id<0||int64_t(id)*bs>query) continue;
        int start=id*bs;
        int valid=min(bs,query+1-start);
        if(group==16)__syncthreads();
        if(group==7) {
            for(int j=lane;j<bs;j+=32) {
                float dot=-INFINITY;
                if(active && j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    dot=0.f;
                    for(int d=0;d<128;++d)
                        dot=fmaf(rounded_q[d][warp],float(k[kb+d]),dot);
                }
                scores[warp][j]=dot;
            }
        } else if(group==16) {
            for(int task=threadIdx.x;task<8*bs;task+=blockDim.x) {
                const int lh=task%8,j=task/8;
                float dot=-INFINITY;
                if(j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    float parts[32]={};
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        #pragma unroll
                        for(int l=0;l<32;++l)
                            parts[l]=fmaf(rounded_q[l+i*32][lh],float(k[kb+l+i*32]),parts[l]);
                    }
                    #pragma unroll
                    for(int delta=16;delta;delta/=2) {
                        #pragma unroll
                        for(int l=0;l<delta;++l)parts[l]+=parts[l+delta];
                    }
                    dot=parts[0];
                }
                scores[lh][j]=dot;
            }
        } else {
        // 2 independent token dots in flight; each retains its original
        // dimension FMA and shuffle reduction order. No additional K/V LDS.
        for(int base=0;base<bs;base+=2) {
            float score[2]={0};
            #pragma unroll
            for(int u=0;u<2;++u) {
                int j=base+u;
                if(active&&j<valid) {
                    int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane;
                    if(group==7) {
                      if(lane==0)for(int d=0;d<128;++d)score[u]=fmaf(float(T(float(q[qo+d])*scale)),float(k[off-lane+d]),score[u]);
                    } else for(int d=0;d<4;++d)score[u]=fmaf(qd[d],float(k[off+d*32]),score[u]);
                } else score[u]=-INFINITY;
            }
            if(group!=7)for(int delta=16;delta;delta/=2)for(int u=0;u<2;++u)score[u]+=__shfl_down_sync(0xffffffff,score[u],delta,32);
            if(lane==0) {
                #pragma unroll
                for(int u=0;u<2;++u) if(base+u<bs) scores[warp][base+u]=score[u];
            }
        }
        }
        if(group==16)__syncthreads();else __syncwarp();
        float localmax=-INFINITY;
        for(int j=lane;j<bs;j+=32) localmax=fmaxf(localmax,scores[warp][j]);
        for(int delta=16;delta;delta/=2) localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
        float nm=fmaxf(maximum,localmax),alpha=expf(maximum-nm),blocksum=0;
        for(int j=lane;j<bs;j+=32) {
            float probability=active?expf(scores[warp][j]-nm):0.f;
            blocksum+=probability;scores[warp][j]=float(T(probability));
        }
        for(int delta=16;delta;delta/=2) blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
        denom=denom*alpha+blocksum;maximum=nm;
        if(lane==0){head_alpha[warp]=alpha;head_denom[warp]=denom;}
        __syncthreads();
        if(pv_active) {
            #pragma unroll
            for(int h=0;h<4;++h)for(int d=0;d<2;++d)acc[h*2+d]*=head_alpha[pv_head+h];
            #pragma unroll 8
            for(int j=0;j<valid;++j) {
                const int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+pv_lane*2;
                const uint32_t packed=*reinterpret_cast<const uint32_t*>(v+off);
                T values[2];__builtin_memcpy(values,&packed,4);
                #pragma unroll
                for(int h=0;h<4;++h) {
                    const float probability=scores[pv_head+h][j];
                    #pragma unroll
                    for(int d=0;d<2;++d)acc[h*2+d]=fmaf(probability,float(values[d]),acc[h*2+d]);
                }
            }
        }
        __syncthreads();
    }
    if(pv_active) {
        #pragma unroll
        for(int h=0;h<4;++h) {
            const int64_t po=((int64_t(batch)*tokens+query)*hq+kv*16+blockIdx.z*8+pv_head+h)*128+pv_lane*2;
            #pragma unroll
            for(int d=0;d<2;++d)out[po+d]=T(acc[h*2+d]/head_denom[pv_head+h]);
        }
    }
    return;
}

extern "C" int personal_nsa_selected_packed_r46_bf16u8(const void* q,const void* k,const void* v,void* scratch,
 const int32_t* ids,void* out,int b,int t,int hq,int hkv,int width,int bs,float scale,int dtype,void* stream) {
    if(dtype==1&&hq/hkv==16&&bs==64&&t>=1024&&(reinterpret_cast<uintptr_t>(v)&7)==0) {
        dim3 grid(b*t,hkv,2);
        nsaSelectedBF16Unroll8Cuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,
            (const __nv_bfloat16*)v,ids,(__nv_bfloat16*)out,t,hq,hkv,width,bs,scale);
        return int(cudaGetLastError());
    }
    return personal_nsa_selected_packed(q,k,v,scratch,ids,out,b,t,hq,hkv,width,bs,scale,dtype,stream);
}

template<class T> __global__ __launch_bounds__(256)
void nsaSelectedBF16Unroll16Cuda(const T* q,const T* k,const T* v,const int32_t* ids,
                              T* out,int tokens,int hq,int hkv,int width,int bs,float scale) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int query=blockIdx.x%tokens,batch=blockIdx.x/tokens,kv=blockIdx.y;
    const int group=16;int gh=blockIdx.z*8+warp,head=kv*group+gh;
    bool active=gh<group;
    const int pv_lane=threadIdx.x%64;
    const int pv_head=(threadIdx.x/64)*4;
    const bool pv_active=pv_head<8;
    int64_t qo=((int64_t(batch)*tokens+query)*hq+head)*128;
    int64_t io=((int64_t(batch)*tokens+query)*hkv+kv)*width;
    __shared__ float scores[8][128];
    __shared__ float rounded_q[128][9];
    __shared__ float head_alpha[8],head_denom[8];
    float qd[4]={0,0,0,0},acc[8]={};
    if(active) for(int d=0;d<4;++d) qd[d]=float(T(float(q[qo+lane+d*32])*scale));
    // G7 preserves the author's dimension-ordered FMA, but assigns
    // independent token dots to lanes. Only Q and scores use shared memory.
    if((group==7 || group==16) && active)
        for(int d=lane;d<128;d+=32)
            rounded_q[d][warp]=float(T(float(q[qo+d])*scale));
    if(lane==0)head_denom[warp]=0.f;
    if(group==16)__syncthreads();else __syncwarp();
    float maximum=-INFINITY,denom=0;
    for(int slot=0;slot<width;++slot) {
        int id=ids[io+slot];
        if(id<0||int64_t(id)*bs>query) continue;
        int start=id*bs;
        int valid=min(bs,query+1-start);
        if(group==16)__syncthreads();
        if(group==7) {
            for(int j=lane;j<bs;j+=32) {
                float dot=-INFINITY;
                if(active && j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    dot=0.f;
                    for(int d=0;d<128;++d)
                        dot=fmaf(rounded_q[d][warp],float(k[kb+d]),dot);
                }
                scores[warp][j]=dot;
            }
        } else if(group==16) {
            for(int task=threadIdx.x;task<8*bs;task+=blockDim.x) {
                const int lh=task%8,j=task/8;
                float dot=-INFINITY;
                if(j<valid) {
                    const int64_t kb=((int64_t(batch)*tokens+start+j)*hkv+kv)*128;
                    float parts[32]={};
                    #pragma unroll
                    for(int i=0;i<4;++i) {
                        #pragma unroll
                        for(int l=0;l<32;++l)
                            parts[l]=fmaf(rounded_q[l+i*32][lh],float(k[kb+l+i*32]),parts[l]);
                    }
                    #pragma unroll
                    for(int delta=16;delta;delta/=2) {
                        #pragma unroll
                        for(int l=0;l<delta;++l)parts[l]+=parts[l+delta];
                    }
                    dot=parts[0];
                }
                scores[lh][j]=dot;
            }
        } else {
        // 2 independent token dots in flight; each retains its original
        // dimension FMA and shuffle reduction order. No additional K/V LDS.
        for(int base=0;base<bs;base+=2) {
            float score[2]={0};
            #pragma unroll
            for(int u=0;u<2;++u) {
                int j=base+u;
                if(active&&j<valid) {
                    int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+lane;
                    if(group==7) {
                      if(lane==0)for(int d=0;d<128;++d)score[u]=fmaf(float(T(float(q[qo+d])*scale)),float(k[off-lane+d]),score[u]);
                    } else for(int d=0;d<4;++d)score[u]=fmaf(qd[d],float(k[off+d*32]),score[u]);
                } else score[u]=-INFINITY;
            }
            if(group!=7)for(int delta=16;delta;delta/=2)for(int u=0;u<2;++u)score[u]+=__shfl_down_sync(0xffffffff,score[u],delta,32);
            if(lane==0) {
                #pragma unroll
                for(int u=0;u<2;++u) if(base+u<bs) scores[warp][base+u]=score[u];
            }
        }
        }
        if(group==16)__syncthreads();else __syncwarp();
        float localmax=-INFINITY;
        for(int j=lane;j<bs;j+=32) localmax=fmaxf(localmax,scores[warp][j]);
        for(int delta=16;delta;delta/=2) localmax=fmaxf(localmax,__shfl_xor_sync(0xffffffff,localmax,delta,32));
        float nm=fmaxf(maximum,localmax),alpha=expf(maximum-nm),blocksum=0;
        for(int j=lane;j<bs;j+=32) {
            float probability=active?expf(scores[warp][j]-nm):0.f;
            blocksum+=probability;scores[warp][j]=float(T(probability));
        }
        for(int delta=16;delta;delta/=2) blocksum+=__shfl_xor_sync(0xffffffff,blocksum,delta,32);
        denom=denom*alpha+blocksum;maximum=nm;
        if(lane==0){head_alpha[warp]=alpha;head_denom[warp]=denom;}
        __syncthreads();
        if(pv_active) {
            #pragma unroll
            for(int h=0;h<4;++h)for(int d=0;d<2;++d)acc[h*2+d]*=head_alpha[pv_head+h];
            #pragma unroll 16
            for(int j=0;j<valid;++j) {
                const int64_t off=((int64_t(batch)*tokens+start+j)*hkv+kv)*128+pv_lane*2;
                const uint32_t packed=*reinterpret_cast<const uint32_t*>(v+off);
                T values[2];__builtin_memcpy(values,&packed,4);
                #pragma unroll
                for(int h=0;h<4;++h) {
                    const float probability=scores[pv_head+h][j];
                    #pragma unroll
                    for(int d=0;d<2;++d)acc[h*2+d]=fmaf(probability,float(values[d]),acc[h*2+d]);
                }
            }
        }
        __syncthreads();
    }
    if(pv_active) {
        #pragma unroll
        for(int h=0;h<4;++h) {
            const int64_t po=((int64_t(batch)*tokens+query)*hq+kv*16+blockIdx.z*8+pv_head+h)*128+pv_lane*2;
            #pragma unroll
            for(int d=0;d<2;++d)out[po+d]=T(acc[h*2+d]/head_denom[pv_head+h]);
        }
    }
    return;
}

extern "C" int personal_nsa_selected_packed_r46_bf16u16(const void* q,const void* k,const void* v,void* scratch,
 const int32_t* ids,void* out,int b,int t,int hq,int hkv,int width,int bs,float scale,int dtype,void* stream) {
    if(dtype==1&&hq/hkv==16&&bs==64&&t>=1024&&(reinterpret_cast<uintptr_t>(v)&7)==0) {
        dim3 grid(b*t,hkv,2);
        nsaSelectedBF16Unroll16Cuda<<<grid,256,0,(cudaStream_t)stream>>>((const __nv_bfloat16*)q,(const __nv_bfloat16*)k,
            (const __nv_bfloat16*)v,ids,(__nv_bfloat16*)out,t,hq,hkv,width,bs,scale);
        return int(cudaGetLastError());
    }
    return personal_nsa_selected_packed(q,k,v,scratch,ids,out,b,t,hq,hkv,width,bs,scale,dtype,stream);
}
