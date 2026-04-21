#ifndef INFINI_OPS_BASE_PAGED_CACHING_H_
#define INFINI_OPS_BASE_PAGED_CACHING_H_

#include "operator.h"

namespace infini::ops {

// Scatters contiguous `(k, v)` tokens into a paged `(k_cache, v_cache)` KV
// cache. `slot_mapping[i]` is the destination slot for token `i`. Negative
// entries mark padding tokens that should be skipped. The destination slot
// `s` maps to `block = s / block_size`, `offset = s % block_size`, where
// `block_size` is the third-to-last dimension of `k_cache` / `v_cache`.
//
// Expected shapes:
//   `k_cache`, `v_cache`: `[num_blocks, num_kv_heads, block_size, head_size]`.
//   `k`, `v`:             `[num_tokens, num_kv_heads, head_size]`.
//   `slot_mapping`:       `[num_tokens]` with integer dtype.
class PagedCaching : public Operator<PagedCaching> {
 public:
  PagedCaching(Tensor k_cache, Tensor v_cache, const Tensor k, const Tensor v,
               const Tensor slot_mapping)
      : k_cache_type_{k_cache.dtype()},
        v_cache_type_{v_cache.dtype()},
        k_type_{k.dtype()},
        v_type_{v.dtype()},
        slot_mapping_type_{slot_mapping.dtype()},
        k_cache_shape_{k_cache.shape()},
        v_cache_shape_{v_cache.shape()},
        k_shape_{k.shape()},
        v_shape_{v.shape()},
        slot_mapping_shape_{slot_mapping.shape()},
        k_cache_strides_{k_cache.strides()},
        v_cache_strides_{v_cache.strides()},
        k_strides_{k.strides()},
        v_strides_{v.strides()},
        slot_mapping_strides_{slot_mapping.strides()} {
    assert(k_cache.ndim() >= 4 && "`k_cache` must have at least 4 dimensions");
    assert(v_cache.ndim() >= 4 && "`v_cache` must have at least 4 dimensions");
    assert(k.ndim() == 3 && "`k` must be 3D `[ntok, nkvh, dh]`");
    assert(v.ndim() == 3 && "`v` must be 3D `[ntok, nkvh, dh]`");
    assert(slot_mapping.ndim() == 1 && "`slot_mapping` must be 1D");
    assert(k_cache.dtype() == v_cache.dtype() &&
           k_cache.dtype() == k.dtype() && k_cache.dtype() == v.dtype() &&
           "`k_cache`, `v_cache`, `k`, and `v` must share the same dtype");
  }

  virtual void operator()(Tensor k_cache, Tensor v_cache, const Tensor k,
                          const Tensor v,
                          const Tensor slot_mapping) const = 0;

 protected:
  const DataType k_cache_type_;

  const DataType v_cache_type_;

  const DataType k_type_;

  const DataType v_type_;

  const DataType slot_mapping_type_;

  Tensor::Shape k_cache_shape_;

  Tensor::Shape v_cache_shape_;

  Tensor::Shape k_shape_;

  Tensor::Shape v_shape_;

  Tensor::Shape slot_mapping_shape_;

  Tensor::Strides k_cache_strides_;

  Tensor::Strides v_cache_strides_;

  Tensor::Strides k_strides_;

  Tensor::Strides v_strides_;

  Tensor::Strides slot_mapping_strides_;
};

}  // namespace infini::ops

#endif
