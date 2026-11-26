#pragma once
#include "operator.h"

namespace infini::ops {
class DeepseekV4C4Index : public Operator<DeepseekV4C4Index> {
 public:
  DeepseekV4C4Index(const Tensor key, const Tensor slots, const Tensor query,
                    const Tensor weights, const Tensor pages, const Tensor lengths,
                    Tensor cache, Tensor out) {}
  virtual void operator()(const Tensor key, const Tensor slots, const Tensor query,
                          const Tensor weights, const Tensor pages, const Tensor lengths,
                          Tensor cache, Tensor out) const = 0;
};
}
