#ifndef INFINI_OPS_KUNLUN_CAUSAL_CONV1D_KERNEL_H_
#define INFINI_OPS_KUNLUN_CAUSAL_CONV1D_KERNEL_H_

#include <stdexcept>

#include "base/causal_conv1d.h"

namespace infini::ops {

template <>
class Operator<CausalConv1d, Device::Type::kKunlun> : public CausalConv1d {
 public:
  using CausalConv1d::CausalConv1d;

  [[noreturn]] void operator()(
      const Tensor qkv, const Tensor weight, Tensor conv_state,
      const std::optional<Tensor> bias, const std::optional<Tensor> cu_seqlens,
      const std::optional<Tensor> initial_state_indices,
      const std::optional<Tensor> final_state_indices, Tensor out,
      std::optional<Tensor> final_conv_state) const override {
    throw std::runtime_error("CausalConv1d is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
