#ifndef INFINI_OPS_TORCH_RESHAPE_AND_CACHE_H_
#define INFINI_OPS_TORCH_RESHAPE_AND_CACHE_H_

#include <string>

#include "base/reshape_and_cache.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<ReshapeAndCache, kDev, 1> : public ReshapeAndCache {
 public:
  Operator(const Tensor key, const Tensor value, Tensor key_cache,
           Tensor value_cache, const Tensor slot_mapping,
           const std::string kv_cache_dtype, const Tensor k_scale,
           const Tensor v_scale);

  void operator()(const Tensor key, const Tensor value, Tensor key_cache,
                  Tensor value_cache, const Tensor slot_mapping,
                  const std::string kv_cache_dtype, const Tensor k_scale,
                  const Tensor v_scale) const override;
};

}  // namespace infini::ops

#endif
