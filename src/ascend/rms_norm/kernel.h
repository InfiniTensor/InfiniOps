#ifndef INFINI_OPS_ASCEND_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ASCEND_RMS_NORM_KERNEL_H_

#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_rms_norm.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/rms_norm.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kAscend> : public RmsNorm {
 public:
  Operator(const Tensor input, const Tensor weight, float eps, Tensor out)
      : RmsNorm(input, weight, eps, out),
        in_cache_(input),
        weight_cache_(weight),
        out_cache_(out) {
    // aclnnRmsNorm writes rstd as a required side output.
    // Allocate a persistent device buffer for it.
    rstd_shape_ = {static_cast<int64_t>(batch_size_),
                   static_cast<int64_t>(nhead_)};
    size_t rstd_bytes = batch_size_ * nhead_ * sizeof(float);
    aclrtMalloc(&rstd_data_, rstd_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

    // The rstd descriptor has a stable data pointer.
    rstd_tensor_ = aclCreateTensor(rstd_shape_.data(), 2, ACL_FLOAT,
                                   /*strides=*/nullptr, 0, ACL_FORMAT_ND,
                                   rstd_shape_.data(), 2, rstd_data_);
  }

  ~Operator() {
    if (executor_) aclDestroyAclOpExecutor(executor_);
    if (rstd_tensor_) aclDestroyTensor(rstd_tensor_);
    if (rstd_data_) aclrtFree(rstd_data_);
  }

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    auto t_in = in_cache_.get(const_cast<void*>(input.data()));
    auto t_weight = weight_cache_.get(const_cast<void*>(weight.data()));
    auto t_out = out_cache_.get(out.data());

    if (!executor_) {
      aclnnRmsNormGetWorkspaceSize(t_in, t_weight, eps, t_out, rstd_tensor_,
                                   &ws_size_, &executor_);
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_in,
                            const_cast<void*>(input.data()));
      aclSetInputTensorAddr(executor_, 1, t_weight,
                            const_cast<void*>(weight.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
      // rstd at output index 1 has a stable address — no update needed.
    }

    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::workspacePool().ensure(stream, ws_size_);
    aclnnRmsNorm(arena.buf, ws_size_, executor_, stream);
  }

 private:
  mutable ascend::AclTensorCache in_cache_;

  mutable ascend::AclTensorCache weight_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;

  std::vector<int64_t> rstd_shape_;

  void* rstd_data_ = nullptr;

  aclTensor* rstd_tensor_ = nullptr;
};

}  // namespace infini::ops

#endif
