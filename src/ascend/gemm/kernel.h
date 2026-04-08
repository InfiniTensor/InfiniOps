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
        beta_val_{beta.value_or(1.0f)} {
    alpha_scalar_ = aclCreateScalar(&alpha_val_, ACL_FLOAT);
    beta_scalar_ = aclCreateScalar(&beta_val_, ACL_FLOAT);
  }

  ~Operator() {
    aclDestroyScalar(alpha_scalar_);
    aclDestroyScalar(beta_scalar_);
  }

  void operator()(const Tensor a, const Tensor b, std::optional<float> alpha,
                  std::optional<float> beta, std::optional<int> trans_a,
                  std::optional<int> trans_b, Tensor c) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    auto t_self = ascend::buildAclTensor(c);
    auto t_a = ascend::buildAclTensor(a, trans_a_);
    auto t_b = ascend::buildAclTensor(b, trans_b_);
    auto t_out = ascend::buildAclTensor(c);

    uint64_t ws_needed = 0;
    aclOpExecutor* executor = nullptr;

    if (batched_) {
      aclnnBaddbmmGetWorkspaceSize(t_self, t_a, t_b, beta_scalar_,
                                   alpha_scalar_, t_out, 0, &ws_needed,
                                   &executor);
    } else {
      aclnnAddmmGetWorkspaceSize(t_self, t_a, t_b, beta_scalar_, alpha_scalar_,
                                 t_out, 0, &ws_needed, &executor);
    }

    auto& arena = ascend::workspacePool().ensure(stream, ws_needed);

    if (batched_) {
      aclnnBaddbmm(arena.buf, ws_needed, executor, stream);
    } else {
      aclnnAddmm(arena.buf, ws_needed, executor, stream);
    }

    aclDestroyTensor(t_self);
    aclDestroyTensor(t_a);
    aclDestroyTensor(t_b);
    aclDestroyTensor(t_out);
  }

 private:
  bool batched_;
  float alpha_val_;
  float beta_val_;
  aclScalar* alpha_scalar_ = nullptr;
  aclScalar* beta_scalar_ = nullptr;
};

}  // namespace infini::ops

#endif
