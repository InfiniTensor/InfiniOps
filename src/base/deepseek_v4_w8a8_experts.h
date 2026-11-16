#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_W8A8_EXPERTS_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_W8A8_EXPERTS_H_

#include "operator.h"

namespace infini::ops {

// Reference W8A8 path: channelwise weights and per-row symmetric activation
// quantization and INT32 dot products. Routed experts round each weighted
// contribution to the activation dtype before FP32 reduction and scaling.
// A null routed_scale selects the shared-expert activation and epilogue.
// Weight tensors are rank-local. TP reduction is the caller's responsibility.
class DeepseekV4W8a8Experts : public Operator<DeepseekV4W8a8Experts> {
 public:
  DeepseekV4W8a8Experts(const Tensor x, const Tensor weights, const Tensor indices,
                        const Tensor gate_up, const Tensor gate_up_scale,
                        const Tensor down, const Tensor down_scale,
                        double swiglu_limit, const std::optional<double> routed_scale, Tensor out) {
    (void)x;
    (void)weights;
    (void)indices;
    (void)gate_up;
    (void)gate_up_scale;
    (void)down;
    (void)down_scale;
    (void)swiglu_limit;
    (void)routed_scale;
    (void)out;
  }

  virtual void operator()(const Tensor x, const Tensor weights, const Tensor indices,
                          const Tensor gate_up, const Tensor gate_up_scale,
                          const Tensor down, const Tensor down_scale,
                          double swiglu_limit, const std::optional<double> routed_scale, Tensor out) const = 0;
};

}  // namespace infini::ops

#endif
