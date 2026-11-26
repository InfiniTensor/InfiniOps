#ifndef INFINI_OPS_BASE_SCATTER_ND_H_
#define INFINI_OPS_BASE_SCATTER_ND_H_

#include "operator.h"

namespace infini::ops {

class ScatterNd : public Operator<ScatterNd> {
 public:
  ScatterNd(const Tensor input, const Tensor indices, const Tensor updates,
            bool add, Tensor out) {}
  virtual void operator()(const Tensor input, const Tensor indices,
                          const Tensor updates, bool add, Tensor out) const = 0;
};

}  // namespace infini::ops

#endif
