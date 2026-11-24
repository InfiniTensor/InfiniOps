#pragma once
#include "base/nsa_compress_paged_cache.h"
#include "arithmetic.cuh"
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include <cuda_runtime.h>
#include <limits>
namespace infini::ops {
namespace hygon_nsa_cache {
template<class I,class T>
__global__ void Launch(T* ko,T* vo,const T* kc,const T* vc,const I* bt,const I* lens,
    size_t width,size_t page,size_t sub,int block,int tail,
    ptrdiff_t kob,ptrdiff_t koh,ptrdiff_t vob,ptrdiff_t voh,
    ptrdiff_t kcb,ptrdiff_t kch,ptrdiff_t kcr,ptrdiff_t vcb,ptrdiff_t vch,ptrdiff_t vcr,
    ptrdiff_t btb,ptrdiff_t ls) {
  op::nsa_compress_paged_cache::hygon::compressPagedCacheKernel<I,T>(
      ko,vo,kc,vc,bt,lens,width,page,sub,block,tail,kob,koh,vob,voh,kcb,kch,kcr,vcb,vch,vcr,btb,ls);
}
}
template<>
class Operator<NsaCompressPagedCache,Device::Type::kHygon> : public NsaCompressPagedCache {
 public:
  using NsaCompressPagedCache::NsaCompressPagedCache;
  static bool supports(const Tensor kc,const Tensor vc,const Tensor bt,const Tensor lens,
      int block,bool tail,Tensor ko,Tensor vo) {
    return kc.device().type()==Device::Type::kHygon &&
      NsaCompressPagedCache::supports(kc,vc,bt,lens,block,tail,ko,vo) &&
      bt.size(0)<=std::numeric_limits<unsigned>::max() && kc.size(1)<=65535 &&
      bt.size(1)<=size_t(65535)*size_t(block)/kc.size(2);
  }
  bool capture_safe() const override {return true;}
  void operator()(const Tensor kc,const Tensor vc,const Tensor bt,const Tensor lens,
      int block,bool tail,Tensor ko,Tensor vo) const override {
    Check(kc,vc,bt,lens,block,tail,ko,vo);
    if(!supports(kc,vc,bt,lens,block,tail,ko,vo))throw std::invalid_argument("Unsupported Hygon NSA cache launch dimensions");
    if(!bt.size(0))return;
    DispatchFunc<List<DataType::kFloat16,DataType::kBFloat16>,List<DataType::kInt32,DataType::kInt64,DataType::kUInt32>>(
      {int64_t(kc.dtype()),int64_t(bt.dtype())},[&](auto tag){
        using T=TypeMapType<Device::Type::kHygon,ListGet<0>(tag)>;using I=TypeMapType<Device::Type::kHygon,ListGet<1>(tag)>;
        const size_t max_blocks=tail?1:bt.size(1)*(kc.size(2)/size_t(block));
        hygon_nsa_cache::Launch<I,T><<<dim3(bt.size(0),max_blocks,kc.size(1)),128,0,static_cast<cudaStream_t>(stream_)>>>(
          static_cast<T*>(ko.data()),static_cast<T*>(vo.data()),static_cast<const T*>(kc.data()),static_cast<const T*>(vc.data()),static_cast<const I*>(bt.data()),static_cast<const I*>(lens.data()),
          bt.size(1),kc.size(2),kc.size(2)/size_t(block),block,int(tail),
          ko.stride(0),ko.stride(1),vo.stride(0),vo.stride(1),kc.stride(0),kc.stride(1),kc.stride(2),vc.stride(0),vc.stride(1),vc.stride(2),bt.stride(0),lens.stride(0));
      },"HygonNsaCompressPagedCache");
    const auto result=cudaGetLastError();if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
  }
};
}
