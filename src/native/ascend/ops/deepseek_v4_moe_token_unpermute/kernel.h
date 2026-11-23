#ifndef INFINI_OPS_ASCEND_DEEPSEEK_V4_MOE_TOKEN_UNPERMUTE_KERNEL_H_
#define INFINI_OPS_ASCEND_DEEPSEEK_V4_MOE_TOKEN_UNPERMUTE_KERNEL_H_

#include <stdexcept>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_moe_token_unpermute.h"
#include "base/deepseek_v4_moe_token_unpermute.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<DeepseekV4MoeTokenUnpermute, Device::Type::kAscend, 0>
    : public DeepseekV4MoeTokenUnpermute {
 public:
  static bool supports(Tensor out, const Tensor permuted_tokens,
                       const Tensor sorted_indices, const Tensor probs) {
    return permuted_tokens.device().type() == Device::Type::kAscend &&
           permuted_tokens.dtype() == DataType::kBFloat16 &&
           permuted_tokens.ndim() == 2 && permuted_tokens.IsContiguous() &&
           sorted_indices.device() == permuted_tokens.device() &&
           sorted_indices.dtype() == DataType::kInt32 &&
           sorted_indices.IsContiguous() &&
           probs.device() == permuted_tokens.device() &&
           probs.dtype() == DataType::kBFloat16 && probs.ndim() == 2 &&
           probs.IsContiguous() && out.device() == permuted_tokens.device() &&
           out.dtype() == DataType::kBFloat16 && out.ndim() == 2 &&
           out.IsContiguous();
  }

  Operator(Tensor out, const Tensor permuted_tokens,
           const Tensor sorted_indices, const Tensor probs)
      : DeepseekV4MoeTokenUnpermute(
            out, permuted_tokens, sorted_indices, probs),
        out_cache_(out),
        permuted_tokens_cache_(permuted_tokens),
        sorted_indices_cache_(sorted_indices),
        probs_cache_(probs) {
    if (!supports(out, permuted_tokens, sorted_indices, probs)) {
      throw std::invalid_argument(
          "unsupported Ascend DeepseekV4MoeTokenUnpermute operands");
    }
    t_out_ = out_cache_.get(out.data());
    t_permuted_tokens_ = permuted_tokens_cache_.get(
        const_cast<void*>(permuted_tokens.data()));
    t_sorted_indices_ =
        sorted_indices_cache_.get(const_cast<void*>(sorted_indices.data()));
    t_probs_ = probs_cache_.get(const_cast<void*>(probs.data()));
    ascend::CheckAcl(aclnnMoeTokenUnpermuteGetWorkspaceSize(
                         t_permuted_tokens_, t_sorted_indices_, t_probs_,
                         false, nullptr, t_out_, &workspace_size_, &executor_),
                     "DeepseekV4MoeTokenUnpermute workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                     "DeepseekV4MoeTokenUnpermute repeatable executor");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    out_cache_.release();
    permuted_tokens_cache_.release();
    sorted_indices_cache_.release();
    probs_cache_.release();
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor out, const Tensor permuted_tokens,
                  const Tensor sorted_indices,
                  const Tensor probs) const override {
    auto* t_out = out_cache_.get(out.data());
    auto* t_permuted_tokens = permuted_tokens_cache_.get(
        const_cast<void*>(permuted_tokens.data()));
    auto* t_sorted_indices =
        sorted_indices_cache_.get(const_cast<void*>(sorted_indices.data()));
    auto* t_probs = probs_cache_.get(const_cast<void*>(probs.data()));
    ascend::CheckAcl(aclSetInputTensorAddr(
                         executor_, 0, t_permuted_tokens,
                         const_cast<void*>(permuted_tokens.data())),
                     "DeepseekV4MoeTokenUnpermute tokens address update");
    ascend::CheckAcl(aclSetInputTensorAddr(
                         executor_, 1, t_sorted_indices,
                         const_cast<void*>(sorted_indices.data())),
                     "DeepseekV4MoeTokenUnpermute indices address update");
    ascend::CheckAcl(aclSetInputTensorAddr(
                         executor_, 2, t_probs,
                         const_cast<void*>(probs.data())),
                     "DeepseekV4MoeTokenUnpermute probs address update");
    ascend::CheckAcl(aclSetOutputTensorAddr(executor_, 0, t_out, out.data()),
                     "DeepseekV4MoeTokenUnpermute output address update");
    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size_, "deepseek_v4_moe_token_unpermute");
    ascend::CheckAcl(aclnnMoeTokenUnpermute(
                         arena.buf, workspace_size_, executor_, stream),
                     "DeepseekV4MoeTokenUnpermute execution");
  }

 private:
  mutable ascend::AclTensorCache out_cache_;
  mutable ascend::AclTensorCache permuted_tokens_cache_;
  mutable ascend::AclTensorCache sorted_indices_cache_;
  mutable ascend::AclTensorCache probs_cache_;
  aclTensor* t_out_{nullptr};
  aclTensor* t_permuted_tokens_{nullptr};
  aclTensor* t_sorted_indices_{nullptr};
  aclTensor* t_probs_{nullptr};
  mutable aclOpExecutor* executor_{nullptr};
  uint64_t workspace_size_{0};
};

}  // namespace infini::ops

#endif
