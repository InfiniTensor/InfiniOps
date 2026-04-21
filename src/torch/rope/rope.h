#ifndef INFINI_OPS_TORCH_ROPE_H_
#define INFINI_OPS_TORCH_ROPE_H_

#include "base/rope.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Rope, kDev, 1> : public Rope {
 public:
  Operator(const Tensor x, const Tensor positions, const Tensor sin_cache,
           const Tensor cos_cache, bool is_neox_style, Tensor out);

  void operator()(const Tensor x, const Tensor positions,
                  const Tensor sin_cache, const Tensor cos_cache,
                  bool is_neox_style, Tensor out) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif
