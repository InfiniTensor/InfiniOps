#ifndef INFINI_OPS_ASCEND_LINEAR_KERNEL_H_
#define INFINI_OPS_ASCEND_LINEAR_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_addmm.h"
#include "aclnnop/aclnn_baddbmm.h"
#include "aclnnop/aclnn_matmul.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/linear.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kAscend> : public Linear {
 public:
  Operator(const Tensor a, const Tensor b, std::optional<Tensor> bias,
           bool trans_a, bool trans_b, Tensor out)
      : Linear(a, b, bias, trans_a, trans_b, out),
        batched_{out.ndim() > 2},
        a_cache_(a, trans_a),
        b_cache_(b, trans_b),
        out_cache_(out) {
    if (has_bias_) {
      bias_cache_ = ascend::AclTensorCache(*bias);
      alpha_scalar_ = aclCreateScalar(&alpha_storage_, ACL_FLOAT);
      beta_scalar_ = aclCreateScalar(&beta_storage_, ACL_FLOAT);
    }
  }

  ~Operator() {
    if (executor_) aclDestroyAclOpExecutor(executor_);
    if (alpha_scalar_) aclDestroyScalar(alpha_scalar_);
    if (beta_scalar_) aclDestroyScalar(beta_scalar_);
  }

  void operator()(const Tensor a, const Tensor b, std::optional<Tensor> bias,
                  bool trans_a, bool trans_b, Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_a = a_cache_.get(const_cast<void*>(a.data()));
    auto t_b = b_cache_.get(const_cast<void*>(b.data()));
    auto t_out = out_cache_.get(out.data());

    if (has_bias_) {
      auto t_bias = bias_cache_.get(const_cast<void*>(bias->data()));

      if (!executor_) {
        if (batched_) {
          aclnnBaddbmmGetWorkspaceSize(t_bias, t_a, t_b, beta_scalar_,
                                       alpha_scalar_, t_out, 0, &ws_size_,
                                       &executor_);
        } else {
          aclnnAddmmGetWorkspaceSize(t_bias, t_a, t_b, beta_scalar_,
                                     alpha_scalar_, t_out, 0, &ws_size_,
                                     &executor_);
        }
        aclSetAclOpExecutorRepeatable(executor_);
      } else {
        aclSetInputTensorAddr(executor_, 0, t_bias,
                              const_cast<void*>(bias->data()));
        aclSetInputTensorAddr(executor_, 1, t_a,
                              const_cast<void*>(a.data()));
        aclSetInputTensorAddr(executor_, 2, t_b,
                              const_cast<void*>(b.data()));
        aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
      }

      auto& arena = ascend::workspacePool().ensure(stream, ws_size_);

      if (batched_) {
        aclnnBaddbmm(arena.buf, ws_size_, executor_, stream);
      } else {
        aclnnAddmm(arena.buf, ws_size_, executor_, stream);
      }
    } else {
      if (!executor_) {
        int8_t cube_math_type = 1;
        aclnnMatmulGetWorkspaceSize(t_a, t_b, t_out, cube_math_type,
                                    &ws_size_, &executor_);
        aclSetAclOpExecutorRepeatable(executor_);
      } else {
        aclSetInputTensorAddr(executor_, 0, t_a,
                              const_cast<void*>(a.data()));
        aclSetInputTensorAddr(executor_, 1, t_b,
                              const_cast<void*>(b.data()));
        aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
      }

      auto& arena = ascend::workspacePool().ensure(stream, ws_size_);
      aclnnMatmul(arena.buf, ws_size_, executor_, stream);
    }
  }

 private:
  bool batched_;

  mutable ascend::AclTensorCache bias_cache_;

  mutable ascend::AclTensorCache a_cache_;

  mutable ascend::AclTensorCache b_cache_;

  mutable ascend::AclTensorCache out_cache_;

  float alpha_storage_ = 1.0f;

  float beta_storage_ = 1.0f;

  aclScalar* alpha_scalar_ = nullptr;

  aclScalar* beta_scalar_ = nullptr;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;
};

}  // namespace infini::ops

#endif
