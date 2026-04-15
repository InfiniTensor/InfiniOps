#ifndef INFINI_OPS_ASCEND_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ASCEND_ADD_RMS_NORM_KERNEL_H_

#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_add.h"
#include "aclnn_rms_norm.h"
#include "ascend/add_rms_norm/registry.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

// Decomposed implementation: aclnnAdd + aclnnRmsNorm.
//
// The fused aclnnAddRmsNorm API has ~200 us host-side launch overhead that
// dominates small-tensor dispatch.  Decomposing into two fast ACLNN calls
// reduces host dispatch from ~224 us to ~56 us (4x faster) with negligible
// NPU-side impact for inference tensor sizes.
template <>
class Operator<AddRmsNorm, Device::Type::kAscend, 0> : public AddRmsNorm {
 public:
  Operator(const Tensor x1, const Tensor x2, const Tensor gamma, float eps,
           Tensor y_out, Tensor x_out)
      : AddRmsNorm(x1, x2, gamma, eps, y_out, x_out),
        x1_cache_(x1),
        x2_cache_(x2),
        gamma_cache_(gamma),
        y_out_cache_(y_out),
        x_out_cache_(x_out) {
    // Alpha scalar for aclnnAdd (x_out = x1 + 1.0 * x2).
    alpha_ = aclCreateScalar(&alpha_storage_, ACL_FLOAT);

    // aclnnRmsNorm writes rstd as a required side output.
    // Size computed here; buffer obtained from pool in `operator()`.
    rstd_shape_ = {static_cast<int64_t>(batch_size_),
                   static_cast<int64_t>(nhead_)};
    rstd_size_ = batch_size_ * nhead_ * sizeof(float);
  }

  ~Operator() {
    if (add_exec_) aclDestroyAclOpExecutor(add_exec_);
    if (norm_exec_) aclDestroyAclOpExecutor(norm_exec_);
    aclDestroyScalar(alpha_);
    if (rstd_tensor_) aclDestroyTensor(rstd_tensor_);
  }

  void operator()(const Tensor x1, const Tensor x2, const Tensor gamma,
                  float eps, Tensor y_out, Tensor x_out) const override {
    auto t_x1 = x1_cache_.get(const_cast<void*>(x1.data()));
    auto t_x2 = x2_cache_.get(const_cast<void*>(x2.data()));
    auto t_gamma = gamma_cache_.get(const_cast<void*>(gamma.data()));
    auto t_y_out = y_out_cache_.get(y_out.data());
    auto t_x_out = x_out_cache_.get(x_out.data());
    auto stream = static_cast<aclrtStream>(stream_);

    // Step 1: x_out = x1 + x2.
    if (!add_exec_) {
      aclnnAddGetWorkspaceSize(t_x1, t_x2, alpha_, t_x_out, &add_ws_,
                               &add_exec_);
      aclSetAclOpExecutorRepeatable(add_exec_);
    } else {
      aclSetInputTensorAddr(add_exec_, 0, t_x1, const_cast<void*>(x1.data()));
      aclSetInputTensorAddr(add_exec_, 1, t_x2, const_cast<void*>(x2.data()));
      aclSetOutputTensorAddr(add_exec_, 0, t_x_out, x_out.data());
    }
    auto& add_arena = ascend::workspacePool().ensure(stream, add_ws_);
    aclnnAdd(add_arena.buf, add_ws_, add_exec_, stream);

    // Obtain shared rstd buffer from pool.
    auto& rstd_arena =
        ascend::workspacePool().ensure(stream, rstd_size_, "temp");

    // Lazily create rstd tensor descriptor on first call.
    if (!rstd_tensor_) {
      rstd_tensor_ = aclCreateTensor(rstd_shape_.data(), 2, ACL_FLOAT,
                                     /*strides=*/nullptr, 0, ACL_FORMAT_ND,
                                     rstd_shape_.data(), 2, rstd_arena.buf);
    } else {
      aclSetRawTensorAddr(rstd_tensor_, rstd_arena.buf);
    }

    // Step 2: y_out = rms_norm(x_out, gamma, eps).
    if (!norm_exec_) {
      aclnnRmsNormGetWorkspaceSize(t_x_out, t_gamma, eps, t_y_out, rstd_tensor_,
                                   &norm_ws_, &norm_exec_);
      aclSetAclOpExecutorRepeatable(norm_exec_);
    } else {
      aclSetInputTensorAddr(norm_exec_, 0, t_x_out, x_out.data());
      aclSetInputTensorAddr(norm_exec_, 1, t_gamma,
                            const_cast<void*>(gamma.data()));
      aclSetOutputTensorAddr(norm_exec_, 0, t_y_out, y_out.data());
      aclSetOutputTensorAddr(norm_exec_, 1, rstd_tensor_, rstd_arena.buf);
    }
    auto& norm_arena = ascend::workspacePool().ensure(stream, norm_ws_);
    aclnnRmsNorm(norm_arena.buf, norm_ws_, norm_exec_, stream);
  }

 private:
  mutable ascend::AclTensorCache x1_cache_;

  mutable ascend::AclTensorCache x2_cache_;

  mutable ascend::AclTensorCache gamma_cache_;

  mutable ascend::AclTensorCache y_out_cache_;

  mutable ascend::AclTensorCache x_out_cache_;

  float alpha_storage_ = 1.0f;

  aclScalar* alpha_ = nullptr;

  std::vector<int64_t> rstd_shape_;

  uint64_t rstd_size_ = 0;

  mutable aclTensor* rstd_tensor_ = nullptr;

  mutable aclOpExecutor* add_exec_ = nullptr;

  mutable uint64_t add_ws_ = 0;

  mutable aclOpExecutor* norm_exec_ = nullptr;

  mutable uint64_t norm_ws_ = 0;
};

}  // namespace infini::ops

#endif
