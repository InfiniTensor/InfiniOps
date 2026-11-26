#include <mma.h>

// Four native CUDA WMMA warps share 16 query rows and a 64-key tile.
// Inactive extra warps on a 32-lane target still participate in CTA barriers.
template<class T> __global__ __launch_bounds__(256)
void nsaCompressionWmmaCuda(const T* q,const T* k,const T* v,T* out,
                             int tokens,int keytokens,int hq,int hkv,int bs,float* lse) {
    using namespace nvcuda;
    constexpr int DS=136,NS=72; // Padding avoids power-of-two shared-memory strides.
    int tid=threadIdx.x,warp=tid/warpSize,query0=blockIdx.x*16,batch=blockIdx.y,head=blockIdx.z;
    int kv=head/(hq/hkv);
    __shared__ __align__(32) T qs[16*DS],tiles[64*DS],ps[16*NS];
    T* ks=tiles;T* vs=tiles; // K is no longer needed when P*V begins.
    __shared__ __align__(32) float scores[16*NS],acc[16*DS];
    __shared__ float maximum[16],denom[16],rescale[16];
    for(int i=tid;i<16*128;i+=256) {
        int row=query0+i/128,d=i%128;
        qs[(i/128)*DS+d]=row<tokens?q[((int64_t(batch)*tokens+row)*hq+head)*128+d]:T(0.f);
        acc[(i/128)*DS+d]=0.f;
    }
    if(tid<16) {maximum[tid]=-INFINITY;denom[tid]=0;}
    __syncthreads();
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
        for(int i=tid;i<16*128;i+=256) acc[(i/128)*DS+i%128]*=rescale[i/128];
        __syncthreads();
        if(warp<4) {
            for(int col=warp*16;col<128;col+=64) {
                wmma::fragment<wmma::accumulator,16,16,16,float> c;
                wmma::load_matrix_sync(c,acc+col,DS,wmma::mem_row_major);
                for(int j=0;j<64;j+=16) {
                    wmma::fragment<wmma::matrix_a,16,16,16,T,wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b,16,16,16,T,wmma::row_major> b;
                    wmma::load_matrix_sync(a,ps+j,NS);
                    wmma::load_matrix_sync(b,vs+j*DS+col,DS);
                    wmma::mma_sync(c,a,b,c);
                }
                wmma::store_matrix_sync(acc+col,c,DS,wmma::mem_row_major);
            }
        }
        __syncthreads();
    }
    for(int i=tid;i<16*128;i+=256) {
        int row=query0+i/128,d=i%128;
        if(row<tokens) out[((int64_t(batch)*tokens+row)*hq+head)*128+d]=T(acc[(i/128)*DS+d]/denom[i/128]);
    }
    if(lse && tid<16 && query0+tid<tokens) lse[(int64_t(batch)*hq+head)*tokens+query0+tid]=(maximum[tid]+log2f(denom[tid]))*0.6931471805599453f;
    return;
}
