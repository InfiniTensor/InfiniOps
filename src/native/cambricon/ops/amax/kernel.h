#ifndef INFINI_OPS_CAMBRICON_AMAX_KERNEL_H_
#define INFINI_OPS_CAMBRICON_AMAX_KERNEL_H_

#include "base/amax.h"
#include "native/cambricon/reduce_cnnl.h"

namespace infini::ops {

template <>
class Operator<Amax, Device::Type::kCambricon> : public Amax {
 public:
  Operator(const Tensor input, const std::vector<int64_t> dim, const bool keepdim,
           Tensor out)
      : Amax{input, dim, keepdim, out},
        plan_{CNNL_REDUCE_MAX, input, dim, keepdim, out} {}

  std::size_t workspace_size_in_bytes() const override { return plan_.WorkspaceSize(); }

  void operator()(const Tensor input, const std::vector<int64_t>, const bool,
                  Tensor out) const override {
    plan_.Run(input, out, static_cast<cnrtQueue_t>(stream_),
              workspace_, workspace_size_in_bytes_);
  }

 private:
  cnnl_utils::ReducePlan plan_;
};

}  // namespace infini::ops

#endif
