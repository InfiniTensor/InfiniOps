#ifndef INFINI_OPS_TORCH_CAST_H_
#define INFINI_OPS_TORCH_CAST_H_
#include "base/cast.h"
#include "torch/tensor_bounds.h"
namespace infini::ops {
template <Device::Type kDev>
class Operator<Cast, kDev, 8> : public Cast {
 public:
  static bool supports(const Tensor input, Tensor out) {
    return input.device().type() == kDev && input.device() == out.device() &&
           input.shape() == out.shape() &&
           (torch_detail::SameView(input, out) ||
            torch_detail::Disjoint(input, out));
  }
  Operator(const Tensor input, Tensor out) : Cast{input, out} {}
  void operator()(const Tensor input, Tensor out) const override;
};
}  // namespace infini::ops
#endif
