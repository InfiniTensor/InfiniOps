#ifndef INFINI_OPS_CUDA_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_CUDA_ROTARY_EMBEDDING_KERNEL_H_

#include <cstdint>
#include <optional>
#include <type_traits>

#include "base/rotary_embedding.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/rotary_embedding/kernel.cuh"

namespace infini::ops {

template <typename Backend>
class CudaRotaryEmbedding : public RotaryEmbedding {
 public:
  using RotaryEmbedding::RotaryEmbedding;

  void operator()(const Tensor positions, Tensor query,
                  std::optional<Tensor> key, const Tensor cos_sin_cache,
                  int64_t, bool, int64_t, bool) const override {
    if (num_tokens_ == 0) {
      return;
    }

    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    bool head_blocks = false;
    if constexpr (Backend::kDeviceType == Device::Type::kIluvatar) {
      if (key && query_type_ == DataType::kBFloat16 &&
          cos_sin_cache_type_ == DataType::kBFloat16 && num_heads_ == 28 &&
          num_kv_heads_ == 28 && rot_dim_ == 128 && rope_dim_offset_ == 0 &&
          num_tokens_ <= 32768 && query_head_stride_ == 128 &&
          key_head_stride_ == 128 &&
          (query_token_stride_ == 3584 || query_token_stride_ == 10752) &&
          (key_token_stride_ == 3584 || key_token_stride_ == 10752)) {
        const auto q = reinterpret_cast<uintptr_t>(query.data());
        const auto k = reinterpret_cast<uintptr_t>(key->data());
        const auto q_span =
            ((num_tokens_ - 1) * query_token_stride_ + 3584) * 2;
        const auto k_span = ((num_tokens_ - 1) * key_token_stride_ + 3584) * 2;
        const bool disjoint = q <= k ? k - q >= q_span : q - k >= k_span;
        const bool packed_qkv = query_token_stride_ == 10752 &&
                                key_token_stride_ == 10752 && q <= k &&
                                k - q == 7168;
        head_blocks = disjoint || packed_qkv;
      }
    }
    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>,
                 ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>>(
        {static_cast<int64_t>(query_type_),
         static_cast<int64_t>(cos_sin_cache_type_)},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using TCache =
              TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
          auto key_data =
              key.has_value() ? reinterpret_cast<T*>(key->data()) : nullptr;
          auto launch = [&](auto neox_tag) {
            constexpr bool kIsNeox = decltype(neox_tag)::value;
            if (head_blocks) {
              RotaryEmbeddingKernel<Backend::kDeviceType, T, TCache, kIsNeox,
                                    true>
                  <<<dim3(num_tokens_, num_heads_), 64, 0, cuda_stream>>>(
                      reinterpret_cast<const int64_t*>(positions.data()),
                      reinterpret_cast<T*>(query.data()), key_data,
                      reinterpret_cast<const TCache*>(cos_sin_cache.data()),
                      cos_sin_cache_strides_[0], query_token_stride_,
                      key_token_stride_, query_head_stride_, key_head_stride_,
                      num_heads_, num_kv_heads_, rot_dim_, rope_dim_offset_,
                      inverse_);
              return;
            }
            RotaryEmbeddingKernel<Backend::kDeviceType, T, TCache, kIsNeox>
                <<<static_cast<uint32_t>(num_tokens_), 256, 0, cuda_stream>>>(
                    reinterpret_cast<const int64_t*>(positions.data()),
                    reinterpret_cast<T*>(query.data()), key_data,
                    reinterpret_cast<const TCache*>(cos_sin_cache.data()),
                    cos_sin_cache_strides_[0], query_token_stride_,
                    key_token_stride_, query_head_stride_, key_head_stride_,
                    num_heads_, num_kv_heads_, rot_dim_, rope_dim_offset_,
                    inverse_);
          };

          if (is_neox_) {
            launch(std::true_type{});
          } else {
            launch(std::false_type{});
          }
        },
        "CudaRotaryEmbedding::operator()");
  }
};

}  // namespace infini::ops

#endif
