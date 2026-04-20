#ifndef INFINI_OPS_TORCH_EMBEDDING_H_
#define INFINI_OPS_TORCH_EMBEDDING_H_

#include "base/embedding.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Embedding, kDev, 1> : public Embedding {
 public:
  Operator(const Tensor input, const Tensor weight, Tensor out);

  void operator()(const Tensor input, const Tensor weight,
                  Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
