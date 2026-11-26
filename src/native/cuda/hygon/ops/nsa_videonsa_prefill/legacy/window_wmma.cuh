#include <mma.h>
__device__ float halfRound(float x){volatile unsigned short b=__half_as_ushort(__float2half_rn(x));return __half2float(__ushort_as_half(b));}

// Four native CUDA WMMA warps share 16 query rows and a 64-key tile.
// Inactive extra warps on a 32-lane target still participate in CTA barriers.
template<class T> __global__ __launch_bounds__(256)
void nsaPrefillWindowRegisterCuda(const T* q,const T* k,const T* v,T* out,
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
            if(row<tokens)out[((int64_t(batch)*tokens+row)*hq+head)*128+d]=T(__bfloat162float(__ushort_as_bfloat16((unsigned short)((__float_as_uint(scores[(i/64)*NS+i%64]/denom[i/64]*1.00390625f)+0x8000u)>>16))));
        }
        __syncthreads();
    }
    return;
}
