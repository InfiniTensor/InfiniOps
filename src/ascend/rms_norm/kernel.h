#ifndef INFINI_OPS_ASCEND_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ASCEND_RMS_NORM_KERNEL_H_

#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_rms_norm.h"
#include "ascend/device.h"
#include "base/rms_norm.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kAscend> : public RmsNorm {
 public:
  Operator(const Tensor input, const Tensor weight, float eps, Tensor out)
      : RmsNorm(input, weight, eps, out) {
    // aclnnRmsNorm writes rstd as a required side output.
    // Allocate a persistent device buffer for it.
    rstd_shape_ = {static_cast<int64_t>(batch_size_),
                   static_cast<int64_t>(nhead_)};
    size_t rstd_bytes = batch_size_ * nhead_ * sizeof(float);
    aclrtMalloc(&rstd_data_, rstd_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
  }

  ~Operator() {
    if (workspace_size_ > 0) aclrtFree(workspace_);
    if (rstd_data_) aclrtFree(rstd_data_);
  }

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    auto t_in     = ascend::buildAclTensor(input);
    auto t_weight = ascend::buildAclTensor(weight);
    auto t_out    = ascend::buildAclTensor(out);
    // rstd is always float32 regardless of input dtype.
    auto t_rstd   = aclCreateTensor(
        rstd_shape_.data(), 2, ACL_FLOAT,
        /*strides=*/nullptr, 0, ACL_FORMAT_ND,
        rstd_shape_.data(), 2, rstd_data_);
    uint64_t ws_needed = 0;
    aclOpExecutor* executor = nullptr;
    aclnnRmsNormGetWorkspaceSize(t_in, t_weight, eps, t_out, t_rstd,
                                 &ws_needed, &executor);
    auto stream = static_cast<aclrtStream>(stream_);
    ascend::ensureWorkspace(workspace_, workspace_size_, ws_needed, stream);
    aclnnRmsNorm(workspace_, workspace_size_, executor, stream);
    aclDestroyTensor(t_in);
    aclDestroyTensor(t_weight);
    aclDestroyTensor(t_out);
    aclDestroyTensor(t_rstd);
  }

 private:
  std::vector<int64_t> rstd_shape_;
  void*            rstd_data_      = nullptr;
  mutable void*    workspace_      = nullptr;
  mutable uint64_t workspace_size_ = 0;
};

}  // namespace infini::ops

#endif
