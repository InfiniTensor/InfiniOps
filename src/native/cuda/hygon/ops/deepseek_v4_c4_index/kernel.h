#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "base/deepseek_v4_c4_index.h"

namespace infini::ops {
namespace v4_c4 {
constexpr size_t kWidth = 128, kPage = 64, kPageBytes = kPage*(kWidth+sizeof(float));
constexpr size_t kTop = 512;

static __device__ float decode(uint8_t bits) {
  const int exp = (bits >> 3) & 15, mantissa = bits & 7;
  const float mag = exp == 0 ? ldexpf(float(mantissa),-9)
                            : ldexpf(1.0f + float(mantissa)/8,exp-7);
  return (bits & 128) ? -mag : mag;
}

static __global__ void transform(const __nv_bfloat16* in, const int32_t* slots, uint8_t* dst,
                                 float* scales, size_t count, size_t pages, size_t stride) {
  __shared__ float lane_value[kWidth];
  const unsigned lane = threadIdx.x;
  for (size_t row=blockIdx.x; row<count; row+=gridDim.x) {
    const int32_t slot = slots ? slots[row] : 0;
    if (slots && (slot < 0 || size_t(slot)/kPage >= pages)) continue;
    float value = __bfloat162float(in[row*kWidth+lane]);
    for (unsigned bit=1; bit<kWidth; bit*=2) {
      lane_value[lane] = value;
      __syncthreads();
      const float other = lane_value[lane ^ bit];
      value = (lane & bit) ? other-value : value+other;
      __syncthreads();
    }
    value *= 0.08838834764831845f;
    lane_value[lane] = fabsf(value);
    __syncthreads();
    for (unsigned bit=kWidth/2; bit; bit/=2) {
      if (lane < bit) lane_value[lane] = fmaxf(lane_value[lane],lane_value[lane+bit]);
      __syncthreads();
    }
    const float scale = fmaxf(lane_value[0],1.0e-4f) / 448.f;
    auto* page = slots ? dst+(size_t(slot)/kPage)*stride : nullptr;
    auto* bytes = slots ? page+(size_t(slot)%kPage)*kWidth : dst+row*kWidth;
    bytes[lane] = __nv_cvt_float_to_fp8(value/scale,__NV_SATFINITE,__NV_E4M3);
    if (lane == 0) {
      if (slots) reinterpret_cast<float*>(page+kPage*kWidth)[size_t(slot)%kPage] = scale;
      else scales[row] = scale;
    }
    __syncthreads();
  }
}

static __device__ int32_t physical(const int32_t* pages, size_t pos, size_t page_count) {
  const auto page = pages[pos/kPage];
  return page < 0 || size_t(page) >= page_count ? -1 : page*int(kPage)+int(pos%kPage);
}

static __global__ void score(const uint8_t* query, const float* qs, const __nv_bfloat16* weights,
                             const int32_t* table, const int32_t* lengths, const uint8_t* cache,
                             float* scores, size_t rows, size_t heads, size_t width,
                             size_t cache_pages, size_t stride) {
  __shared__ float dot[kWidth];
  const unsigned lane=threadIdx.x;
  const size_t span=width*kPage;
  for (size_t flat=blockIdx.x; flat<rows*span; flat+=gridDim.x) {
    const size_t row=flat/span, pos=flat%span;
    if (lengths[row] <= int(kTop) || pos >= size_t(lengths[row])) continue;
    const auto id = physical(table+row*width,pos,cache_pages);
    if (id < 0) {
      if (lane == 0) scores[flat]=-INFINITY;
      continue;
    }
    const auto* page = cache+(size_t(id)/kPage)*stride;
    const auto offset=size_t(id)%kPage;
    const float key=decode(page[offset*kWidth+lane]);
    const float key_scale=reinterpret_cast<const float*>(page+kPage*kWidth)[offset];
    float total=0;
    for (size_t h=0;h<heads;++h) {
      const size_t head=row*heads+h;
      dot[lane]=key*decode(query[head*kWidth+lane]);
      __syncthreads();
      for (unsigned step=kWidth/2;step;step/=2) {
        if (lane<step) dot[lane]+=dot[lane+step];
        __syncthreads();
      }
      if (lane == 0)
        total+=fmaxf(dot[0]*key_scale*qs[head],0.f)*__bfloat162float(weights[head]);
      __syncthreads();
    }
    if (lane == 0) scores[flat]=total;
  }
}

static __global__ void select(float* scores, const int32_t* table, const int32_t* lengths,
                              int32_t* out, size_t rows, size_t width, size_t cache_pages) {
  const size_t span=width*kPage;
  for (size_t row=blockIdx.x*blockDim.x+threadIdx.x;row<rows;row+=gridDim.x*blockDim.x) {
    const size_t length=lengths[row] > 0 ? min(size_t(lengths[row]),span) : 0;
    for (size_t k=0;k<kTop;++k) {
      int32_t idx=-1;
      if (length<=kTop) {
        if (k<length) idx=physical(table+row*width,k,cache_pages);
      } else {
        float best=-INFINITY;
        size_t chosen=0;
        for (size_t j=0;j<length;++j) {
          const float value=scores[row*span+j];
          if (value>best) { best=value; chosen=j; }
        }
        if (best!=-INFINITY) {
          idx=physical(table+row*width,chosen,cache_pages);
          scores[row*span+chosen]=-INFINITY;
        }
      }
      out[row*kTop+k]=idx;
    }
  }
}
} // namespace v4_c4

template <>
class Operator<DeepseekV4C4Index, Device::Type::kHygon> : public DeepseekV4C4Index {
 public:
  static bool supports(Tensor k, Tensor slots, Tensor q, Tensor w, Tensor table,
                       Tensor lens, Tensor cache, Tensor out) {
    using namespace v4_c4;
    if (q.ndim()!=3 || q.size(2)!=kWidth || !q.size(0) || !q.size(1) ||
        table.ndim()!=2 || table.size(0)!=q.size(0) || !table.size(1) ||
        cache.ndim()!=2 || !cache.size(0) || cache.size(0)>INT32_MAX/kPage ||
        cache.size(1)<kPageBytes || cache.size(1)%sizeof(float)) return false;
    const auto rows=q.size(0), heads=q.size(1);
    if (k.shape()!=Tensor::Shape{rows,kWidth} || w.shape()!=Tensor::Shape{rows,heads} ||
        slots.shape()!=Tensor::Shape{rows} || lens.shape()!=Tensor::Shape{rows} ||
        out.shape()!=Tensor::Shape{rows,kTop} || k.dtype()!=DataType::kBFloat16 ||
        q.dtype()!=DataType::kBFloat16 || w.dtype()!=DataType::kBFloat16 ||
        slots.dtype()!=DataType::kInt32 || lens.dtype()!=DataType::kInt32 ||
        table.dtype()!=DataType::kInt32 || out.dtype()!=DataType::kInt32 ||
        cache.dtype()!=DataType::kUInt8 || q.device().type()!=Device::Type::kHygon) return false;
    for (const auto& t : {k,slots,q,w,table,lens,cache,out})
      if (t.device()!=q.device() || !t.IsContiguous()) return false;
    return table.size(1)<=INT32_MAX/kPage &&
           rows<=SIZE_MAX/2/(heads*(kWidth+sizeof(float))) &&
           rows<=SIZE_MAX/2/(table.size(1)*kPage*sizeof(float));
  }
  Operator(Tensor k,Tensor slots,Tensor q,Tensor w,Tensor table,Tensor lens,Tensor cache,Tensor out)
      : DeepseekV4C4Index(k,slots,q,w,table,lens,cache,out) {
    if (!supports(k,slots,q,w,table,lens,cache,out))
      throw std::invalid_argument("Invalid DeepSeek V4 C4 index geometry");
    bytes_=q.numel()+sizeof(float)*(w.numel()+q.size(0)*table.size(1)*v4_c4::kPage);
  }
  bool capture_safe() const override { return true; }
  size_t workspace_size_in_bytes() const override { return bytes_; }
  void operator()(Tensor k,Tensor slots,Tensor q,Tensor w,Tensor table,Tensor lens,Tensor cache,Tensor out) const override {
    using namespace v4_c4;
    if (!supports(k,slots,q,w,table,lens,cache,out) || !workspace_ || workspace_size_in_bytes_<bytes_)
      throw std::invalid_argument("Invalid DeepSeek V4 C4 index execution");
    const auto rows=q.size(0), heads=q.size(1), width=table.size(1);
    const auto grid=[](size_t n){return static_cast<unsigned>(std::min<size_t>(n,65535));};
    auto stream=static_cast<cudaStream_t>(stream_);
    auto* scores=static_cast<float*>(workspace_);
    auto* scales=scores+rows*width*kPage;
    auto* query=reinterpret_cast<uint8_t*>(scales+rows*heads);
    auto* raw=static_cast<uint8_t*>(cache.data());
    const auto* lengths=static_cast<const int32_t*>(lens.data());
    const auto* page_table=static_cast<const int32_t*>(table.data());
    transform<<<grid(rows),kWidth,0,stream>>>(static_cast<const __nv_bfloat16*>(k.data()),
        static_cast<const int32_t*>(slots.data()),raw,nullptr,rows,cache.size(0),cache.size(1));
    if (width*kPage>kTop) {
      transform<<<grid(rows*heads),kWidth,0,stream>>>(
          static_cast<const __nv_bfloat16*>(q.data()),nullptr,query,scales,rows*heads,0,0);
      score<<<grid(rows*width*kPage),kWidth,0,stream>>>(query,scales,
          static_cast<const __nv_bfloat16*>(w.data()),page_table,lengths,raw,scores,
          rows,heads,width,cache.size(0),cache.size(1));
    }
    select<<<grid((rows+63)/64),64,0,stream>>>(scores,page_table,lengths,
        static_cast<int32_t*>(out.data()),rows,width,cache.size(0));
    if (cudaGetLastError()!=cudaSuccess) throw std::runtime_error("DeepSeek V4 C4 index launch failed");
  }
 private:
  size_t bytes_{};
};
} // namespace infini::ops
