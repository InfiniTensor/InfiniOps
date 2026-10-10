#ifndef INFINI_OPS_KUNLUN_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_KUNLUN_ROTARY_EMBEDDING_KERNEL_H_

#include <xpu/runtime.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "base/rotary_embedding.h"
#include "dispatcher.h"
#include "native/kunlun/data_type_.h"
#include "native/kunlun/ops/rotary_embedding/launch.h"

namespace infini::ops {

template <>
class Operator<RotaryEmbedding, Device::Type::kKunlun>
    : public RotaryEmbedding {
 public:
  Operator(const Tensor positions, Tensor query, std::optional<Tensor> key,
           const Tensor cos_sin_cache, int64_t head_size, bool is_neox,
           int64_t rope_dim_offset = 0, bool inverse = false)
      : RotaryEmbedding{positions,       query,     key,
                        cos_sin_cache,   head_size, is_neox,
                        rope_dim_offset, inverse} {
    assert(cos_sin_cache.size(0) > 0 && "RoPE requires a nonempty cache");
    uint64_t available = 0;
    if (xpu_device_get_attr(&available, XPUATTR_NUM_CLUSTER, device_index_) ==
            XPU_SUCCESS &&
        available > 0) {
      cluster_count_ = static_cast<int>(std::min<uint64_t>(available, 12));
    }
  }

  void operator()(const Tensor positions, Tensor query,
                  std::optional<Tensor> key, const Tensor cos_sin_cache,
                  int64_t, bool, int64_t, bool) const override {
    if (num_tokens_ == 0) {
      return;
    }

    const auto queue = static_cast<XPUStream>(stream_ ? stream_ : 0);
    const size_t sequence_length =
        positions_ndim_ == 2 ? positions_shape_[1] : num_tokens_;
    const ptrdiff_t query_batch_stride =
        positions_ndim_ == 2 ? query_strides_[0] : 0;
    const ptrdiff_t key_batch_stride =
        positions_ndim_ == 2 && key.has_value() ? key_strides_[0] : 0;

    DispatchFunc<
        Device::Type::kKunlun,
        List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32>,
        List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32>>(
        {query_type_, cos_sin_cache_type_},
        [&](auto query_tag, auto cache_tag) {
          using T = typename decltype(query_tag)::type;
          using TCache = typename decltype(cache_tag)::type;

          LaunchRotaryEmbeddingKunlun<T, TCache>(
              queue, cluster_count_,
              reinterpret_cast<const int64_t*>(positions.data()),
              reinterpret_cast<T*>(query.data()),
              reinterpret_cast<const TCache*>(cos_sin_cache.data()),
              cos_sin_cache_shape_[0], num_tokens_, sequence_length, num_heads_,
              rot_dim_, rope_dim_offset_, query_batch_stride,
              query_token_stride_, query_head_stride_,
              cos_sin_cache_strides_[0], is_neox_, inverse_);

          if (key.has_value()) {
            LaunchRotaryEmbeddingKunlun<T, TCache>(
                queue, cluster_count_,
                reinterpret_cast<const int64_t*>(positions.data()),
                reinterpret_cast<T*>(key->data()),
                reinterpret_cast<const TCache*>(cos_sin_cache.data()),
                cos_sin_cache_shape_[0], num_tokens_, sequence_length,
                num_kv_heads_, rot_dim_, rope_dim_offset_, key_batch_stride,
                key_token_stride_, key_head_stride_, cos_sin_cache_strides_[0],
                is_neox_, inverse_);
          }
        },
        "KunlunRotaryEmbedding::operator()");
  }

 private:
  int cluster_count_{8};
};

}  // namespace infini::ops

#endif
