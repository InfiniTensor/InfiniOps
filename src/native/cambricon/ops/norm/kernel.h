#ifndef INFINI_OPS_CAMBRICON_NORM_KERNEL_H_
#define INFINI_OPS_CAMBRICON_NORM_KERNEL_H_

#include "base/norm.h"
#include "native/cambricon/reduce_cnnl.h"

namespace infini::ops {

template <>
class Operator<Norm, Device::Type::kCambricon> : public Norm {
 public:
  Operator(const Tensor input, std::optional<double> p,
           std::vector<int64_t> dim, bool keepdim, Tensor out)
      : Norm{input, p, dim, keepdim, out},
        plan_{ValidateMode(p), input, dim, keepdim, out} {}

  std::size_t workspace_size_in_bytes() const override { return plan_.WorkspaceSize(); }

  void operator()(const Tensor input, std::optional<double>,
                  std::vector<int64_t>, bool, Tensor out) const override {
    plan_.Run(input, out, static_cast<cnrtQueue_t>(stream_),
              workspace_, workspace_size_in_bytes_);
  }

 private:
  static cnnlReduceOp_t ValidateMode(std::optional<double> p) {
    if (p.value_or(2.0) != 2.0)
      throw std::invalid_argument("Cambricon Norm supports L2 only");
    return CNNL_REDUCE_NORM2;
  }

  cnnl_utils::ReducePlan plan_;
};

}  // namespace infini::ops

#endif
