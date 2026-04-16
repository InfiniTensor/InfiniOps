#ifndef INFINI_OPS_ASCEND_GEMM_KERNEL_H_
#define INFINI_OPS_ASCEND_GEMM_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_addmm.h"
#include "aclnnop/aclnn_baddbmm.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/gemm.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kAscend> : public Gemm {
 public:
  Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
           std::optional<float> beta, std::optional<int> trans_a,
           std::optional<int> trans_b, Tensor c)
      : Gemm(a, b, alpha, beta, trans_a, trans_b, c),
        batched_{batch_count_ > 1},
        alpha_val_{alpha.value_or(1.0f)},
        beta_val_{beta.value_or(1.0f)},
        self_cache_(c),
        a_cache_(a, trans_a_),
        b_cache_(b, trans_b_),
        out_cache_(c) {
    alpha_scalar_ = aclCreateScalar(&alpha_val_, ACL_FLOAT);
    beta_scalar_ = aclCreateScalar(&beta_val_, ACL_FLOAT);
  }

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;

    // Release tensor caches — executors destroy their tensors internally.
    self_cache_.release();
    a_cache_.release();
    b_cache_.release();
    out_cache_.release();

    if (alpha_scalar_) aclDestroyScalar(alpha_scalar_);
    if (beta_scalar_) aclDestroyScalar(beta_scalar_);
  }

  void operator()(const Tensor a, const Tensor b, std::optional<float> alpha,
                  std::optional<float> beta, std::optional<int> trans_a,
                  std::optional<int> trans_b, Tensor c) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    auto t_self = self_cache_.get(c.data());
    auto t_a = a_cache_.get(const_cast<void*>(a.data()));
    auto t_b = b_cache_.get(const_cast<void*>(b.data()));
    auto t_out = out_cache_.get(c.data());

    if (!executor_) {
      if (batched_) {
        aclnnBaddbmmGetWorkspaceSize(t_self, t_a, t_b, beta_scalar_,
                                     alpha_scalar_, t_out, 0, &ws_size_,
                                     &executor_);
      } else {
        aclnnAddmmGetWorkspaceSize(t_self, t_a, t_b, beta_scalar_,
                                   alpha_scalar_, t_out, 0, &ws_size_,
                                   &executor_);
      }
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_self, c.data());
      aclSetInputTensorAddr(executor_, 1, t_a, const_cast<void*>(a.data()));
      aclSetInputTensorAddr(executor_, 2, t_b, const_cast<void*>(b.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, c.data());
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);

    if (batched_) {
      aclnnBaddbmm(arena.buf, ws_size_, executor_, stream);
    } else {
      aclnnAddmm(arena.buf, ws_size_, executor_, stream);
    }
  }

 private:
  bool batched_;

  float alpha_val_;

  float beta_val_;

  aclScalar* alpha_scalar_ = nullptr;

  aclScalar* beta_scalar_ = nullptr;

  mutable ascend::AclTensorCache self_cache_;

  mutable ascend::AclTensorCache a_cache_;

  mutable ascend::AclTensorCache b_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;
};

}  // namespace infini::ops

#endif
