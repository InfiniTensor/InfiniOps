#pragma once
#include "base/nsa_paged_attention.h"
#ifndef ENABLE_HYGON_API
#define INFINI_NSA_LOCAL_HYGON_MACRO
#define ENABLE_HYGON_API 1
#endif
#include "arithmetic.cuh"
#ifdef INFINI_NSA_LOCAL_HYGON_MACRO
#undef ENABLE_HYGON_API
#undef INFINI_NSA_LOCAL_HYGON_MACRO
#endif
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include <cuda_runtime.h>
#include <limits>
namespace infini::ops {
namespace hygon_nsa_attention {
template <typename Tindex, typename Tdata, typename Tgate>
__global__ void launchNsaPagedDecodeHygonAddressHd128(
    Tdata *out,
    const Tdata *q,
    const Tdata *k_cmp,
    const Tdata *v_cmp,
    const Tdata *k_cache,
    const Tdata *v_cache,
    const Tindex *block_tables,
    const Tindex *cache_lens,
    const Tgate *gates,
    size_t num_heads,
    size_t num_kv_heads,
    float scale,
    size_t max_num_blocks_per_seq,
    size_t page_block_size,
    size_t subblocks_per_page,
    int nsa_block_size,
    int window_size,
    int select_blocks,
    ptrdiff_t q_stride,
    ptrdiff_t q_head_stride,
    ptrdiff_t k_cmp_block_stride,
    ptrdiff_t k_cmp_head_stride,
    ptrdiff_t v_cmp_block_stride,
    ptrdiff_t v_cmp_head_stride,
    ptrdiff_t k_batch_stride,
    ptrdiff_t k_head_stride,
    ptrdiff_t k_row_stride,
    ptrdiff_t v_batch_stride,
    ptrdiff_t v_head_stride,
    ptrdiff_t v_row_stride,
    ptrdiff_t o_stride,
    ptrdiff_t o_head_stride,
    ptrdiff_t block_table_batch_stride,
    ptrdiff_t cache_lens_stride,
    ptrdiff_t gates_seq_stride,
    ptrdiff_t gates_branch_stride,
    ptrdiff_t gates_head_stride) {
    op::nsa_paged_attention::hygon::nsaPagedDecodeHd128Kernel<Tindex, Tdata, Tgate>(
        out, q, k_cmp, v_cmp, k_cache, v_cache, block_tables, cache_lens, gates, num_heads, num_kv_heads, scale,
        max_num_blocks_per_seq, page_block_size, subblocks_per_page, nsa_block_size, window_size, select_blocks, q_stride, q_head_stride,
        k_cmp_block_stride, k_cmp_head_stride, v_cmp_block_stride, v_cmp_head_stride,
        k_batch_stride, k_head_stride, k_row_stride, v_batch_stride, v_head_stride, v_row_stride,
        o_stride, o_head_stride, block_table_batch_stride, cache_lens_stride, gates_seq_stride,
        gates_branch_stride, gates_head_stride);
}


}
template<>
class Operator<NsaPagedAttention,Device::Type::kHygon> : public NsaPagedAttention {
 public:
  using NsaPagedAttention::NsaPagedAttention;
  static bool supports(const Tensor q,const Tensor km,const Tensor vm,const Tensor kc,const Tensor vc,const Tensor bt,const Tensor lens,const Tensor gates,
      float scale,int block,int window,int selected,Tensor out) {
    return q.device().type()==Device::Type::kHygon && NsaPagedAttention::supports(q,km,vm,kc,vc,bt,lens,gates,scale,block,window,selected,out) &&
      q.size(0)<=std::numeric_limits<unsigned>::max() && q.size(1)<=65535 &&
      bt.size(1)<=size_t(std::numeric_limits<int>::max())/kc.size(2);
  }
  bool capture_safe() const override {return true;}
  void operator()(const Tensor q,const Tensor km,const Tensor vm,const Tensor kc,const Tensor vc,const Tensor bt,const Tensor lens,const Tensor gates,
      float scale,int block,int window,int selected,Tensor out) const override {
    Check(q,km,vm,kc,vc,bt,lens,gates,scale,block,window,selected,out);
    if(!supports(q,km,vm,kc,vc,bt,lens,gates,scale,block,window,selected,out))throw std::invalid_argument("Unsupported Hygon NSA attention launch dimensions");
    if(!q.size(0))return;
    DispatchFunc<List<DataType::kFloat16,DataType::kBFloat16>,List<DataType::kInt32,DataType::kInt64,DataType::kUInt32>,List<DataType::kFloat16,DataType::kBFloat16,DataType::kFloat32>>(
      {int64_t(q.dtype()),int64_t(bt.dtype()),int64_t(gates.dtype())},[&](auto tag){
        using T=TypeMapType<Device::Type::kHygon,ListGet<0>(tag)>;
        using I=TypeMapType<Device::Type::kHygon,ListGet<1>(tag)>;
        using G=TypeMapType<Device::Type::kHygon,ListGet<2>(tag)>;
        hygon_nsa_attention::launchNsaPagedDecodeHygonAddressHd128<I,T,G><<<dim3(q.size(0),q.size(1)),128,0,static_cast<cudaStream_t>(stream_)>>>(
          static_cast<T*>(out.data()),static_cast<const T*>(q.data()),static_cast<const T*>(km.data()),static_cast<const T*>(vm.data()),static_cast<const T*>(kc.data()),static_cast<const T*>(vc.data()),static_cast<const I*>(bt.data()),static_cast<const I*>(lens.data()),static_cast<const G*>(gates.data()),
          q.size(1),kc.size(1),scale,bt.size(1),kc.size(2),kc.size(2)/size_t(block),block,window,selected,
          q.stride(0),q.stride(1),km.stride(0),km.stride(1),vm.stride(0),vm.stride(1),kc.stride(0),kc.stride(1),kc.stride(2),vc.stride(0),vc.stride(1),vc.stride(2),out.stride(0),out.stride(1),bt.stride(0),lens.stride(0),gates.stride(0),gates.stride(1),gates.stride(2));
      },"HygonNsaPagedAttention");
    const auto result=cudaGetLastError();if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));
  }
};
}
