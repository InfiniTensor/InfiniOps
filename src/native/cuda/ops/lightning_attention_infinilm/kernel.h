#ifndef INFINI_OPS_CUDA_LIGHTNING_ATTENTION_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_LIGHTNING_ATTENTION_INFINILM_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "base/lightning_attention_infinilm.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/lightning_attention_infinilm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

using LightningAttentionInfinilmDataTypes =
    ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>;

using LightningAttentionInfinilmIndexTypes =
    List<DataType::kInt32, DataType::kInt64>;

template <typename Backend>
class CudaLightningAttentionInfinilm : public LightningAttentionInfinilm {
 public:
  using LightningAttentionInfinilm::LightningAttentionInfinilm;

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor slope, Tensor initial_state,
                  const Tensor initial_state_indices,
                  const Tensor final_state_indices, Tensor out) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    // One thread per state column, so `head_dim` has to fit into one block.
    assert(head_dim_ > 0 &&
           static_cast<int>(head_dim_) <= BackendMaxBlockSize<Backend>::value &&
           "`LightningAttentionInfinilm` requires head_dim to fit one block");
    assert(batch_size_ <= 65535 &&
           "`LightningAttentionInfinilm` requires batch_size <= 65535");

    dim3 grid(static_cast<unsigned>(num_heads_),
              static_cast<unsigned>(batch_size_));
    dim3 block(static_cast<unsigned>(head_dim_));
    const size_t shared_bytes = 2 * head_dim_ * sizeof(float);

    DispatchFunc<LightningAttentionInfinilmDataTypes,
                 LightningAttentionInfinilmIndexTypes>(
        {static_cast<int64_t>(out.dtype()), static_cast<int64_t>(index_dtype_)},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using TIndex =
              TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;

          LightningAttentionInfinilmKernel<Backend::kDeviceType, T, TIndex>
              <<<grid, block, shared_bytes, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()),
                  reinterpret_cast<T*>(initial_state.data()),
                  reinterpret_cast<const T*>(q.data()),
                  reinterpret_cast<const T*>(k.data()),
                  reinterpret_cast<const T*>(v.data()),
                  static_cast<const float*>(slope.data()),
                  reinterpret_cast<const TIndex*>(initial_state_indices.data()),
                  reinterpret_cast<const TIndex*>(final_state_indices.data()),
                  seq_len_, head_dim_, state_pool_stride_, state_head_stride_,
                  state_row_stride_, q_batch_stride_, q_seq_stride_,
                  q_head_stride_, k_batch_stride_, k_seq_stride_, k_head_stride_,
                  v_batch_stride_, v_seq_stride_, v_head_stride_,
                  out_batch_stride_, out_seq_stride_, out_head_stride_,
                  slope_stride_);
        },
        "CudaLightningAttentionInfinilm::operator()");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_LIGHTNING_ATTENTION_INFINILM_KERNEL_H_
