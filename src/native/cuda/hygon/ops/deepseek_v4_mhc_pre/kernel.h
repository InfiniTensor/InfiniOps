#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "base/deepseek_v4_mhc_pre.h"
#include "head.cuh"

namespace infini::ops {
namespace v4_mhc {
constexpr unsigned kThreads=128;
constexpr unsigned kStreams=4, kWidth=24;
constexpr unsigned kSplits = 32;

inline bool use_split_projection(size_t rows, size_t hidden, int64_t iterations) {
  return iterations && rows <= 1024 && (hidden == 4096 || hidden == 7168);
}

inline size_t workspace_floats(size_t rows, size_t hidden, size_t outputs, int64_t iterations) {
  return use_split_projection(rows, hidden, iterations) ?
      rows * kSplits * (outputs + 1) : rows * outputs;
}

// Hygon FP32 MMAC assigns the K chunks to four 16-lane groups and writes
// four output columns per lane. Preserve the reduction order across K splits.
using Float2 = float __attribute__((ext_vector_type(2)));
using Float4 = float __attribute__((ext_vector_type(4)));

template <bool Pack8>
static __global__ void project_split(const __nv_bfloat16* x, const float* w, float* dots,
                                     float* squares, size_t rows, size_t width) {
  const unsigned lane = threadIdx.x;
  const unsigned group = lane / 16;
  const size_t row = blockIdx.x * 16 + lane % 16;
  const size_t col = blockIdx.y * 16 + lane % 16;
  const size_t split_width = width / kSplits;
  const size_t split_start = blockIdx.z * split_width;
  Float4 dot = {};
  float square = 0.0f;
  for (size_t tile = 0; tile < split_width; tile += 64) {
    for (unsigned j = 0; j < 16; j += Pack8 ? 2 : 1) {
      const size_t k = split_start + tile + (j / 8 * 4 + group) * 8 + j % 8;
      const float a = row < rows ? __bfloat162float(x[row * width + k]) : 0.0f;
      const float b = col < kWidth ? w[col * width + k] : 0.0f;
      square = fmaf(a, a, square);
      if constexpr (Pack8) {
        const float a_next = row < rows ? __bfloat162float(x[row * width + k + 1]) : 0.0f;
        const float b_next = col < kWidth ? w[col * width + k + 1] : 0.0f;
        square = fmaf(a_next, a_next, square);
        dot = __builtin_hcu_mmac_16x16x8_f32(Float2{a, a_next}, Float2{b, b_next}, dot);
      } else {
        dot = __builtin_hcu_mmac_16x16x4_f32(a, b, dot);
      }
    }
  }
  const float pair = square + __shfl_xor(square, 32, 64);
  const float total = pair + __shfl_xor(pair, 16, 64);
  if (row >= rows) return;
  const size_t index = blockIdx.z * rows + row;
  if (blockIdx.y == 0 && group == 0) squares[index] = total;
  for (unsigned i = 0; i < 4; ++i) {
    const size_t column = blockIdx.y * 16 + group + i * 4;
    if (column < kWidth) dots[index * kWidth + column] = dot[i];
  }
}

static __device__ float sum_splits(const float* data, size_t stride) {
  float partials[4] = {};
  for (unsigned split = 0; split < kSplits; split += 4) {
    for (unsigned group = 0; group < 4; ++group) {
      partials[group] += data[(split + group) * stride];
    }
  }
  return ((partials[0] + partials[1]) + partials[2]) + partials[3];
}

static __device__ float sum_pairs(float a, float b, float c, float d) {
  return (a + b) + (c + d);
}

static __global__ void project(const __nv_bfloat16* x,const float* w,float* values,
                               size_t rows,size_t width,size_t outputs,float eps) {
  __shared__ float dots[kThreads], squares[kThreads];
  const auto lane=threadIdx.x;
  for(size_t idx=blockIdx.x;idx<rows*outputs;idx+=gridDim.x) {
    const auto row=idx/outputs,col=idx%outputs;
    float dot=0,square=0;
    for(size_t i=lane;i<width;i+=kThreads) {
      const float v=__bfloat162float(x[row*width+i]);
      dot+=v*w[col*width+i];
      square+=v*v;
    }
    dots[lane]=dot; squares[lane]=square;
    __syncthreads();
    for(unsigned step=kThreads/2;step;step/=2) {
      if(lane<step) { dots[lane]+=dots[lane+step]; squares[lane]+=squares[lane+step]; }
      __syncthreads();
    }
    if(lane==0) values[idx]=dots[0]*rsqrtf(squares[0]/float(width)+eps);
    __syncthreads();
  }
}

static __device__ float logistic(float x, bool split) {
  return 1.0f / (1.0f + (split ? __builtin_amdgcn_exp2f(-x * 1.4426950408889634f) : expf(-x)));
}

static __global__ void mix(const __nv_bfloat16* residual,const float* values,const float* scales,
                           const float* bias,__nv_bfloat16* collapsed,float* post,float* combination,
                           size_t rows,size_t hidden,float eps,int64_t iterations,
                           const float* squares,float norm_eps) {
  __shared__ float pre[kStreams];
  const size_t width=iterations ? kWidth : kStreams;
  const bool split = squares != nullptr;
  for(size_t row=blockIdx.x;row<rows;row+=gridDim.x) {
    if(threadIdx.x==0) {
      float projected[kWidth];
      const auto* v=values+row*width;
      if (split) {
        const float inv_rms = rsqrtf(sum_splits(squares + row, rows) / float(kStreams * hidden) + norm_eps);
        for (unsigned i = 0; i < kWidth; ++i) projected[i] = sum_splits(v + i, rows * width) * inv_rms;
        v = projected;
      }
      for(unsigned i=0;i<kStreams;++i) pre[i]=logistic(v[i]*scales[0]+bias[i],split)+eps;
      if(iterations) {
        auto* matrix=combination+row*kStreams*kStreams;
        for(unsigned i=0;i<kStreams;++i) {
          post[row*kStreams+i]=2*logistic(v[kStreams+i]*scales[1]+bias[kStreams+i],split);
          float entries[kStreams],max_value=-INFINITY,sum=0;
          for(unsigned j=0;j<kStreams;++j) {
            const auto idx=2*kStreams+i*kStreams+j;
            entries[j]=v[idx]*scales[2]+bias[idx];
            max_value=fmaxf(max_value,entries[j]);
          }
          for(float &entry:entries) { entry=expf(entry-max_value); sum+=entry; }
          if (split) sum = sum_pairs(entries[0], entries[1], entries[2], entries[3]);
          for(unsigned j=0;j<kStreams;++j) matrix[i*kStreams+j]=entries[j]/sum+eps;
        }
        for(int64_t iter=0;iter<iterations;++iter) {
          if(iter) for(unsigned i=0;i<kStreams;++i) {
            float sum=eps;
            const auto* entries = matrix + i * kStreams;
            if (split) sum = sum_pairs(entries[0], entries[1], entries[2], entries[3]) + eps;
            else for(unsigned j=0;j<kStreams;++j) sum+=entries[j];
            for(unsigned j=0;j<kStreams;++j) matrix[i*kStreams+j]/=sum;
          }
          for(unsigned j=0;j<kStreams;++j) {
            if (split) {
              const float a = matrix[j], b = matrix[kStreams+j];
              const float c = matrix[2*kStreams+j], d = matrix[3*kStreams+j];
              const float even = sum_pairs(a,d,c,b) + eps;
              const float odd = sum_pairs(b,a,d,c) + eps;
              for(unsigned i=0;i<kStreams;++i) matrix[i*kStreams+j]/=i%2 ? odd : even;
            } else {
              float sum=eps;
              for(unsigned i=0;i<kStreams;++i) sum+=matrix[i*kStreams+j];
              for(unsigned i=0;i<kStreams;++i) matrix[i*kStreams+j]/=sum;
            }
          }
        }
      }
    }
    __syncthreads();
    for(size_t i=threadIdx.x;i<hidden;i+=kThreads) {
      if (split) {
        const float a = __bfloat162float(residual[(row*kStreams)*hidden+i]);
        const float b = __bfloat162float(residual[(row*kStreams+1)*hidden+i]);
        const float c = __bfloat162float(residual[(row*kStreams+2)*hidden+i]);
        const float d = __bfloat162float(residual[(row*kStreams+3)*hidden+i]);
        // Each HCU lane pair multiplies its partner first, then uses FMA.
        const float left = fmaf(a,pre[0],b*pre[1]);
        const float right = fmaf(c,pre[2],d*pre[3]);
        collapsed[row*hidden+i] = __float2bfloat16_rz(left+right);
      } else {
        float sum=0;
        for(unsigned h=0;h<kStreams;++h)
          sum+=__bfloat162float(residual[(row*kStreams+h)*hidden+i])*pre[h];
        collapsed[row*hidden+i]=__float2bfloat16(sum);
      }
    }
    __syncthreads();
  }
}
} // namespace v4_mhc

template<>
class Operator<DeepseekV4MhcPre,Device::Type::kHygon> : public DeepseekV4MhcPre {
 public:
  static bool supports(Tensor x,Tensor w,Tensor s,Tensor b,double e,double m,int64_t it,
                       Tensor y,Tensor p,Tensor c) {
    using namespace v4_mhc;
    if(x.ndim()!=3 || x.size(1)!=kStreams || !x.size(0) || !x.size(2) ||
       x.dtype()!=DataType::kBFloat16 || x.device().type()!=Device::Type::kHygon ||
       it<0 || it>1000 || !std::isfinite(e) || e<=0 || !std::isfinite(m) || m<=0) return false;
    const auto n=x.size(0),h=x.size(2);
    const size_t outputs=it ? kWidth : kStreams;
    if(w.shape()!=Tensor::Shape{outputs,kStreams*h} || s.shape()!=Tensor::Shape{it ? 3u : 1u} ||
       b.shape()!=Tensor::Shape{outputs} || y.shape()!=Tensor::Shape{n,h} ||
       p.shape()!=Tensor::Shape{n,kStreams} || c.shape()!=Tensor::Shape{n,kStreams,kStreams} ||
       y.dtype()!=x.dtype()) return false;
    for(const auto& t:{w,s,b,p,c}) if(t.dtype()!=DataType::kFloat32) return false;
    for(const auto& t:{x,w,s,b,y,p,c}) if(t.device()!=x.device() || !t.IsContiguous()) return false;
    return n<=SIZE_MAX/outputs/sizeof(float);
  }
  Operator(Tensor x,Tensor w,Tensor s,Tensor b,double e,double m,int64_t it,Tensor y,Tensor p,Tensor c)
      : DeepseekV4MhcPre(x,w,s,b,e,m,it,y,p,c) {
    if(!supports(x,w,s,b,e,m,it,y,p,c)) throw std::invalid_argument("Invalid DeepSeek V4 mHC pre");
    bytes_=v4_mhc::workspace_floats(x.size(0),x.size(2),w.size(0),it)*sizeof(float);
  }
  bool capture_safe() const override {return true;}
  size_t workspace_size_in_bytes() const override {return bytes_;}
  void operator()(Tensor x,Tensor w,Tensor s,Tensor b,double e,double m,int64_t it,Tensor y,Tensor p,Tensor c) const override {
    using namespace v4_mhc;
    if(!supports(x,w,s,b,e,m,it,y,p,c) || !workspace_ ||
       workspace_size_in_bytes_<workspace_floats(x.size(0),x.size(2),w.size(0),it)*sizeof(float))
      throw std::invalid_argument("Invalid DeepSeek V4 mHC pre execution");
    const auto grid=[](size_t n){return static_cast<unsigned>(std::min<size_t>(n,65535));};
    auto stream=static_cast<cudaStream_t>(stream_);
    auto* v=static_cast<float*>(workspace_);
    const auto* residual=static_cast<const __nv_bfloat16*>(x.data());
    const auto rows = x.size(0);
    const auto width = kStreams*x.size(2);
    const auto* weight = static_cast<const float*>(w.data());
    if (!it && rows <= 65535 && (x.size(2)==4096 || x.size(2)==7168)) {
      v4_mhc_head::head<<<grid(rows),256,0,stream>>>(residual,weight,
          static_cast<const float*>(s.data()),static_cast<const float*>(b.data()),
          static_cast<__nv_bfloat16*>(y.data()),x.size(2),float(e),float(m));
      if(cudaGetLastError()!=cudaSuccess) throw std::runtime_error("DeepSeek V4 mHC head launch failed");
      return;
    }
    float* squares = nullptr;
    if (use_split_projection(rows,x.size(2),it)) {
      squares = v + rows*kSplits*kWidth;
      const dim3 split_grid((rows+15)/16,2,kSplits);
      if (rows > 64) project_split<true><<<split_grid,64,0,stream>>>(residual,weight,v,squares,rows,width);
      else project_split<false><<<split_grid,64,0,stream>>>(residual,weight,v,squares,rows,width);
    } else {
      project<<<grid(rows*w.size(0)),kThreads,0,stream>>>(residual,weight,v,rows,width,w.size(0),float(e));
    }
    if(cudaGetLastError()!=cudaSuccess) throw std::runtime_error("DeepSeek V4 mHC projection launch failed");
    mix<<<grid(x.size(0)),kThreads,0,stream>>>(residual,v,static_cast<const float*>(s.data()),
        static_cast<const float*>(b.data()),static_cast<__nv_bfloat16*>(y.data()),
        static_cast<float*>(p.data()),static_cast<float*>(c.data()),x.size(0),x.size(2),float(m),it,squares,float(e));
    if(cudaGetLastError()!=cudaSuccess) throw std::runtime_error("DeepSeek V4 mHC pre launch failed");
  }
 private:
  size_t bytes_{};
};
} // namespace infini::ops
