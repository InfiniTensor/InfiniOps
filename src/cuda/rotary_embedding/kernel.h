#ifndef INFINI_OPS_CUDA_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_CUDA_ROTARY_EMBEDDING_KERNEL_H_

#include <cassert>
#include <cstdint>

#include "base/rotary_embedding.h"
#include "cuda/kernel_commons.cuh"
#include "cuda/rotary_embedding/kernel.cuh"
#include "cuda/runtime_utils.h"
#include "dispatcher.h"

namespace infini::ops {

template <typename Backend>
class CudaRotaryEmbedding : public RotaryEmbedding {
 public:
  using RotaryEmbedding::RotaryEmbedding;

  void operator()(const Tensor positions, const Tensor query, const Tensor key,
                  const Tensor cos_sin_cache, int64_t head_size,
                  int64_t rotary_dim, bool is_neox_style, Tensor query_out,
                  Tensor key_out) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    uint32_t num_blocks = static_cast<uint32_t>(num_tokens_);
    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    assert(query.dtype() == key.dtype() &&
           "query and key must have the same dtype");

    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>,
                 AllCudaBlockSizes>(
        {static_cast<int64_t>(query.dtype()), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          RotaryEmbeddingKernel<kBlockSize, Backend::kDeviceType, float, T>
              <<<num_blocks, kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(query_out.data()),
                  reinterpret_cast<T*>(key_out.data()),
                  reinterpret_cast<const T*>(query.data()),
                  reinterpret_cast<const T*>(key.data()),
                  reinterpret_cast<const T*>(cos_sin_cache.data()),
                  reinterpret_cast<const int64_t*>(positions.data()),
                  num_heads_, num_kv_heads_, head_size_, rotary_dim_,
                  query_strides_[0], query_strides_[1], key_strides_[0],
                  key_strides_[1], query_out_strides_[0],
                  query_out_strides_[1], key_out_strides_[0],
                  key_out_strides_[1], is_neox_style_);
        },
        "CudaRotaryEmbedding::operator()");
  }
};

}  // namespace infini::ops

#endif
