#ifndef INFINI_OPS_TORCH_PAGED_CACHING_H_
#define INFINI_OPS_TORCH_PAGED_CACHING_H_

#include "base/paged_caching.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<PagedCaching, kDev, 1> : public PagedCaching {
 public:
  Operator(Tensor k_cache, Tensor v_cache, const Tensor k, const Tensor v,
           const Tensor slot_mapping);

  void operator()(Tensor k_cache, Tensor v_cache, const Tensor k,
                  const Tensor v,
                  const Tensor slot_mapping) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
