#ifndef INFINI_OPS_BASE_HC_PRE_H_
#define INFINI_OPS_BASE_HC_PRE_H_

#include <cstdint>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class HcPre : public Operator<HcPre> {
 public:
  static constexpr bool kAutomaticDefault = true;

  HcPre(Tensor y, Tensor post, Tensor comb, const Tensor x,
        const Tensor hc_fn, const Tensor hc_scale, const Tensor hc_base,
        int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps,
        double hc_eps) {}

  virtual void operator()(Tensor y, Tensor post, Tensor comb, const Tensor x,
                          const Tensor hc_fn, const Tensor hc_scale,
                          const Tensor hc_base, int64_t hc_mult,
                          int64_t hc_sinkhorn_iters, double norm_eps,
                          double hc_eps) const = 0;
};

}  // namespace infini::ops

#endif
