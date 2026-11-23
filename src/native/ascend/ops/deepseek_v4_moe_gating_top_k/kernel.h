#ifndef INFINI_OPS_ASCEND_DEEPSEEK_V4_MOE_GATING_TOP_K_KERNEL_H_
#define INFINI_OPS_ASCEND_DEEPSEEK_V4_MOE_GATING_TOP_K_KERNEL_H_

#include <optional>
#include <stdexcept>

#include "base/deepseek_v4_moe_gating_top_k.h"
#include "operator.h"

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_moe_gating_top_k_hash.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#endif

namespace infini::ops {

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
template <>
class Operator<DeepseekV4MoeGatingTopK, Device::Type::kAscend, 0>
    : public DeepseekV4MoeGatingTopK {
 public:
  static bool supports(Tensor topk_weights, Tensor topk_ids,
                       Tensor normalized_scores, const Tensor logits,
                       const std::optional<Tensor> bias,
                       const std::optional<Tensor> input_ids,
                       const std::optional<Tensor> tid2eid, int64_t topk,
                       double routed_scaling_factor) {
    const bool optional_ok =
        (!bias || (bias->device() == logits.device() &&
                   bias->dtype() == logits.dtype() && bias->IsContiguous())) &&
        (!input_ids || (input_ids->device() == logits.device() &&
                        (input_ids->dtype() == DataType::kInt32 ||
                         input_ids->dtype() == DataType::kInt64) &&
                        input_ids->IsContiguous())) &&
        (!tid2eid || (tid2eid->device() == logits.device() &&
                      (tid2eid->dtype() == DataType::kInt32 ||
                       tid2eid->dtype() == DataType::kInt64) &&
                      tid2eid->IsContiguous()));
    return logits.device().type() == Device::Type::kAscend &&
           logits.dtype() == DataType::kFloat32 && logits.ndim() == 2 &&
           logits.IsContiguous() && topk > 0 &&
           routed_scaling_factor > 0.0 && optional_ok &&
           topk_weights.device() == logits.device() &&
           topk_weights.dtype() == logits.dtype() &&
           topk_weights.IsContiguous() &&
           topk_ids.device() == logits.device() &&
           topk_ids.dtype() == DataType::kInt32 &&
           topk_ids.IsContiguous() &&
           normalized_scores.device() == logits.device() &&
           normalized_scores.dtype() == DataType::kFloat32 &&
           normalized_scores.IsContiguous();
  }

  Operator(Tensor topk_weights, Tensor topk_ids, Tensor normalized_scores,
           const Tensor logits, const std::optional<Tensor> bias,
           const std::optional<Tensor> input_ids,
           const std::optional<Tensor> tid2eid, int64_t topk,
           double routed_scaling_factor)
      : DeepseekV4MoeGatingTopK(topk_weights, topk_ids, normalized_scores,
                                logits, bias, input_ids, tid2eid, topk,
                                routed_scaling_factor),
        topk_weights_cache_(topk_weights),
        topk_ids_cache_(topk_ids),
        normalized_scores_cache_(normalized_scores),
        logits_cache_(logits),
        bias_cache_(bias ? ascend::AclTensorCache(*bias)
                         : ascend::AclTensorCache()),
        input_ids_cache_(input_ids ? ascend::AclTensorCache(*input_ids)
                                   : ascend::AclTensorCache()),
        tid2eid_cache_(tid2eid ? ascend::AclTensorCache(*tid2eid)
                               : ascend::AclTensorCache()),
        has_bias_(bias.has_value()),
        has_input_ids_(input_ids.has_value()),
        has_tid2eid_(tid2eid.has_value()),
        topk_(topk),
        routed_scaling_factor_(routed_scaling_factor) {
    if (!supports(topk_weights, topk_ids, normalized_scores, logits, bias,
                  input_ids, tid2eid, topk, routed_scaling_factor)) {
      throw std::invalid_argument(
          "unsupported Ascend DeepseekV4MoeGatingTopK operands");
    }
    t_topk_weights_ = topk_weights_cache_.get(topk_weights.data());
    t_topk_ids_ = topk_ids_cache_.get(topk_ids.data());
    t_normalized_scores_ =
        normalized_scores_cache_.get(normalized_scores.data());
    t_logits_ = logits_cache_.get(const_cast<void*>(logits.data()));
    t_bias_ = has_bias_ ? bias_cache_.get(const_cast<void*>(bias->data()))
                        : nullptr;
    t_input_ids_ = has_input_ids_
                       ? input_ids_cache_.get(
                             const_cast<void*>(input_ids->data()))
                       : nullptr;
    t_tid2eid_ = has_tid2eid_
                     ? tid2eid_cache_.get(const_cast<void*>(tid2eid->data()))
                     : nullptr;
    ascend::CheckAcl(aclnnMoeGatingTopKHashGetWorkspaceSize(
                         t_logits_, t_bias_, t_input_ids_, t_tid2eid_, topk_,
                         1, 1, 1, 0, 2, false, routed_scaling_factor_, 1e-20,
                         t_topk_weights_, t_topk_ids_, t_normalized_scores_,
                         &workspace_size_, &executor_),
                     "DeepseekV4MoeGatingTopK workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                     "DeepseekV4MoeGatingTopK repeatable executor");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    topk_weights_cache_.release();
    topk_ids_cache_.release();
    normalized_scores_cache_.release();
    logits_cache_.release();
    if (has_bias_) bias_cache_.release();
    if (has_input_ids_) input_ids_cache_.release();
    if (has_tid2eid_) tid2eid_cache_.release();
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor topk_weights, Tensor topk_ids,
                  Tensor normalized_scores, const Tensor logits,
                  const std::optional<Tensor> bias,
                  const std::optional<Tensor> input_ids,
                  const std::optional<Tensor> tid2eid, int64_t, double) const override {
    auto* t_logits = logits_cache_.get(const_cast<void*>(logits.data()));
    auto* t_topk_weights = topk_weights_cache_.get(topk_weights.data());
    auto* t_topk_ids = topk_ids_cache_.get(topk_ids.data());
    auto* t_normalized_scores =
        normalized_scores_cache_.get(normalized_scores.data());
    ascend::CheckAcl(aclSetInputTensorAddr(
                         executor_, 0, t_logits,
                         const_cast<void*>(logits.data())),
                     "DeepseekV4MoeGatingTopK logits address update");
    if (has_bias_) {
      auto* t_bias = bias_cache_.get(const_cast<void*>(bias->data()));
      ascend::CheckAcl(aclSetInputTensorAddr(
                           executor_, 1, t_bias,
                           const_cast<void*>(bias->data())),
                       "DeepseekV4MoeGatingTopK bias address update");
    }
    if (has_input_ids_) {
      auto* t_input_ids =
          input_ids_cache_.get(const_cast<void*>(input_ids->data()));
      ascend::CheckAcl(aclSetInputTensorAddr(
                           executor_, 2, t_input_ids,
                           const_cast<void*>(input_ids->data())),
                       "DeepseekV4MoeGatingTopK input_ids address update");
    }
    if (has_tid2eid_) {
      auto* t_tid2eid =
          tid2eid_cache_.get(const_cast<void*>(tid2eid->data()));
      ascend::CheckAcl(aclSetInputTensorAddr(
                           executor_, 3, t_tid2eid,
                           const_cast<void*>(tid2eid->data())),
                       "DeepseekV4MoeGatingTopK tid2eid address update");
    }
    ascend::CheckAcl(aclSetOutputTensorAddr(
                         executor_, 0, t_topk_weights, topk_weights.data()),
                     "DeepseekV4MoeGatingTopK weights address update");
    ascend::CheckAcl(aclSetOutputTensorAddr(
                         executor_, 1, t_topk_ids, topk_ids.data()),
                     "DeepseekV4MoeGatingTopK ids address update");
    ascend::CheckAcl(aclSetOutputTensorAddr(
                         executor_, 2, t_normalized_scores,
                         normalized_scores.data()),
                     "DeepseekV4MoeGatingTopK scores address update");
    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size_, "deepseek_v4_moe_gating_top_k");
    ascend::CheckAcl(aclnnMoeGatingTopKHash(
                         arena.buf, workspace_size_, executor_, stream),
                     "DeepseekV4MoeGatingTopK execution");
  }

 private:
  mutable ascend::AclTensorCache topk_weights_cache_;
  mutable ascend::AclTensorCache topk_ids_cache_;
  mutable ascend::AclTensorCache normalized_scores_cache_;
  mutable ascend::AclTensorCache logits_cache_;
  mutable ascend::AclTensorCache bias_cache_;
  mutable ascend::AclTensorCache input_ids_cache_;
  mutable ascend::AclTensorCache tid2eid_cache_;
  bool has_bias_{false};
  bool has_input_ids_{false};
  bool has_tid2eid_{false};
  int64_t topk_{0};
  double routed_scaling_factor_{1.0};
  aclTensor* t_topk_weights_{nullptr};
  aclTensor* t_topk_ids_{nullptr};
  aclTensor* t_normalized_scores_{nullptr};
  aclTensor* t_logits_{nullptr};
  aclTensor* t_bias_{nullptr};
  aclTensor* t_input_ids_{nullptr};
  aclTensor* t_tid2eid_{nullptr};
  mutable aclOpExecutor* executor_{nullptr};
  uint64_t workspace_size_{0};
};
#else
template <>
class Operator<DeepseekV4MoeGatingTopK, Device::Type::kAscend, 0>
    : public DeepseekV4MoeGatingTopK {
 public:
  template <typename... Args>
  static bool supports(const Args&...) { return false; }
  Operator(Tensor a, Tensor b, Tensor c, const Tensor d,
           const std::optional<Tensor> e, const std::optional<Tensor> f,
           const std::optional<Tensor> g, int64_t h, double i)
      : DeepseekV4MoeGatingTopK(a, b, c, d, e, f, g, h, i) {
    throw std::runtime_error(
        "DeepseekV4MoeGatingTopK requires custom transformer opapi");
  }
  void operator()(Tensor, Tensor, Tensor, const Tensor,
                  const std::optional<Tensor>, const std::optional<Tensor>,
                  const std::optional<Tensor>, int64_t, double) const override {}
};
#endif

}  // namespace infini::ops

#endif
