#ifndef INFINI_OPS_CAMBRICON_SUM_KERNEL_H_
#define INFINI_OPS_CAMBRICON_SUM_KERNEL_H_

#include "base/sum.h"
#include "native/cambricon/reduce_cnnl.h"

namespace infini::ops {

template <>
class Operator<Sum, Device::Type::kCambricon> : public Sum {
 public:
  Operator(const Tensor input, const std::optional<std::vector<int64_t>> dim,
           const bool keepdim, const std::optional<DataType> dtype, Tensor out)
      : Sum{input, dim, keepdim, dtype, out},
        plan_{ValidateMode(input, dtype), input,
              dim.value_or(std::vector<int64_t>{}), keepdim, out} {}

  std::size_t workspace_size_in_bytes() const override { return plan_.WorkspaceSize(); }

  void operator()(const Tensor input, const std::optional<std::vector<int64_t>>,
                  const bool, const std::optional<DataType>, Tensor out) const override {
    plan_.Run(input, out, static_cast<cnrtQueue_t>(stream_),
              workspace_, workspace_size_in_bytes_);
  }

 private:
  static cnnlReduceOp_t ValidateMode(const Tensor& input, std::optional<DataType> dtype) {
    if (dtype && *dtype != input.dtype())
      throw std::invalid_argument("Cambricon Sum does not support mixed output dtype");
    return CNNL_REDUCE_ADD;
  }

  cnnl_utils::ReducePlan plan_;
};

}  // namespace infini::ops

#endif
