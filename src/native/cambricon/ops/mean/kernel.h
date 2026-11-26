#ifndef INFINI_OPS_CAMBRICON_MEAN_KERNEL_H_
#define INFINI_OPS_CAMBRICON_MEAN_KERNEL_H_

#include "base/mean.h"
#include "native/cambricon/reduce_cnnl.h"

namespace infini::ops {

template <>
class Operator<Mean, Device::Type::kCambricon> : public Mean {
 public:
  Operator(const Tensor input, const std::optional<std::vector<int64_t>> dim,
           const bool keepdim, const std::optional<DataType> dtype, Tensor out)
      : Mean{input, dim, keepdim, dtype, out},
        plan_{ValidateMode(input, dtype), input,
              dim.value_or(std::vector<int64_t>{}), keepdim, out} {}

  Operator(const Tensor input, const std::optional<DataType> dtype, Tensor out)
      : Operator{input, std::nullopt, false, dtype, out} {}

  std::size_t workspace_size_in_bytes() const override { return plan_.WorkspaceSize(); }

  void operator()(const Tensor input, const std::optional<std::vector<int64_t>>,
                  const bool, const std::optional<DataType>, Tensor out) const override {
    (*this)(input, std::nullopt, out);
  }

  void operator()(const Tensor input, const std::optional<DataType>, Tensor out) const override {
    plan_.Run(input, out, static_cast<cnrtQueue_t>(stream_),
              workspace_, workspace_size_in_bytes_);
  }

 private:
  static cnnlReduceOp_t ValidateMode(const Tensor& input, std::optional<DataType> dtype) {
    if (dtype && *dtype != input.dtype())
      throw std::invalid_argument("Cambricon Mean does not support mixed output dtype");
    return CNNL_REDUCE_AVG;
  }

  cnnl_utils::ReducePlan plan_;
};

}  // namespace infini::ops

#endif
