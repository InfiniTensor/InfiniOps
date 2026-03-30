#ifndef INFINI_OPS_BASE_RESHAPE_AND_CACHE_H_
#define INFINI_OPS_BASE_RESHAPE_AND_CACHE_H_

#include <cstddef>
#include <vector>

#include "operator.h"

namespace infini::ops {

class ReshapeAndCache : public Operator<ReshapeAndCache> {
 public:
  ReshapeAndCache(
      const Tensor key, const Tensor value,
      const Tensor kv_cache, const Tensor slot_mapping,
      Tensor kv_cache_out)
      : num_tokens_{key.size(0)},
        num_kv_heads_{key.size(1)},
        head_size_{key.size(2)},
        block_size_{kv_cache.size(1)},
        key_shape_{key.shape()},
        value_shape_{value.shape()},
        kv_cache_shape_{kv_cache.shape()},
        slot_mapping_shape_{slot_mapping.shape()},
        key_strides_{key.strides()},
        value_strides_{value.strides()},
        kv_cache_strides_{kv_cache.strides()},
        slot_mapping_strides_{slot_mapping.strides()},
        kv_cache_out_strides_{kv_cache_out.strides()} {
    assert(key.shape() == value.shape() &&
           "`ReshapeAndCache` requires key and value same shape");
    assert(kv_cache.ndim() == 4 &&
           "`ReshapeAndCache` requires kv_cache to be 4D [num_blocks, block_size, num_kv_heads, head_size]");
    assert(slot_mapping.ndim() == 1 &&
           "`ReshapeAndCache` requires slot_mapping to be 1D");
  }

  virtual void operator()(
      const Tensor key, const Tensor value,
      const Tensor kv_cache, const Tensor slot_mapping,
      Tensor kv_cache_out) const = 0;

 protected:
  Tensor::Size num_tokens_{0};

  Tensor::Size num_kv_heads_{0};

  Tensor::Size head_size_{0};

  Tensor::Size block_size_{0};

  Tensor::Shape key_shape_;

  Tensor::Shape value_shape_;

  Tensor::Shape kv_cache_shape_;

  Tensor::Shape slot_mapping_shape_;

  Tensor::Strides key_strides_;

  Tensor::Strides value_strides_;

  Tensor::Strides kv_cache_strides_;

  Tensor::Strides slot_mapping_strides_;

  Tensor::Strides kv_cache_out_strides_;
};

}  // namespace infini::ops

#endif
