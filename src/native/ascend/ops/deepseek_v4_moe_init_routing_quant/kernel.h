#ifndef INFINI_OPS_ASCEND_DEEPSEEK_V4_MOE_INIT_ROUTING_QUANT_KERNEL_H_
#define INFINI_OPS_ASCEND_DEEPSEEK_V4_MOE_INIT_ROUTING_QUANT_KERNEL_H_

#include <stdexcept>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_moe_init_routing_quant_v2.h"
#include "base/deepseek_v4_moe_init_routing_quant.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<DeepseekV4MoeInitRoutingQuant, Device::Type::kAscend, 0>
    : public DeepseekV4MoeInitRoutingQuant {
 public:
  static bool supports(Tensor expanded_x, Tensor expanded_row_idx,
                       Tensor expert_tokens_count,
                       Tensor dynamic_quant_scale, const Tensor x,
                       const Tensor expert_idx) {
    return x.device().type() == Device::Type::kAscend &&
           x.dtype() == DataType::kBFloat16 && x.ndim() == 2 &&
           x.IsContiguous() && expert_idx.device() == x.device() &&
           expert_idx.dtype() == DataType::kInt32 &&
           expert_idx.ndim() == 2 && expert_idx.IsContiguous() &&
           expanded_x.device() == x.device() &&
           expanded_x.dtype() == DataType::kInt8 &&
           expanded_x.IsContiguous() &&
           expanded_row_idx.device() == x.device() &&
           expanded_row_idx.dtype() == DataType::kInt32 &&
           expanded_row_idx.IsContiguous() &&
           expert_tokens_count.device() == x.device() &&
           expert_tokens_count.dtype() == DataType::kInt32 &&
           expert_tokens_count.IsContiguous() &&
           dynamic_quant_scale.device() == x.device() &&
           dynamic_quant_scale.dtype() == DataType::kFloat32 &&
           dynamic_quant_scale.IsContiguous();
  }

  Operator(Tensor expanded_x, Tensor expanded_row_idx,
           Tensor expert_tokens_count, Tensor dynamic_quant_scale,
           const Tensor x, const Tensor expert_idx)
      : DeepseekV4MoeInitRoutingQuant(
            expanded_x, expanded_row_idx, expert_tokens_count,
            dynamic_quant_scale, x, expert_idx),
        expanded_x_cache_(expanded_x),
        expanded_row_idx_cache_(expanded_row_idx),
        expert_tokens_count_cache_(expert_tokens_count),
        dynamic_quant_scale_cache_(dynamic_quant_scale),
        x_cache_(x),
        expert_idx_cache_(expert_idx) {
    if (!supports(expanded_x, expanded_row_idx, expert_tokens_count,
                  dynamic_quant_scale, x, expert_idx)) {
      throw std::invalid_argument(
          "unsupported Ascend DeepseekV4MoeInitRoutingQuant operands");
    }
    t_expanded_x_ = expanded_x_cache_.get(expanded_x.data());
    t_expanded_row_idx_ =
        expanded_row_idx_cache_.get(expanded_row_idx.data());
    t_expert_tokens_count_ =
        expert_tokens_count_cache_.get(expert_tokens_count.data());
    t_dynamic_quant_scale_ =
        dynamic_quant_scale_cache_.get(dynamic_quant_scale.data());
    t_x_ = x_cache_.get(const_cast<void*>(x.data()));
    t_expert_idx_ =
        expert_idx_cache_.get(const_cast<void*>(expert_idx.data()));
    const int64_t active_num = static_cast<int64_t>(expanded_x.shape()[0]);
    const int64_t expert_num =
        static_cast<int64_t>(expert_tokens_count.shape()[0]);
    ascend::CheckAcl(aclnnMoeInitRoutingQuantV2GetWorkspaceSize(
                         t_x_, t_expert_idx_, nullptr, nullptr, active_num, 0,
                         expert_num, 0, 2, true, 1, t_expanded_x_,
                         t_expanded_row_idx_, t_expert_tokens_count_, nullptr,
                         t_dynamic_quant_scale_, &workspace_size_, &executor_),
                     "DeepseekV4MoeInitRoutingQuant workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                     "DeepseekV4MoeInitRoutingQuant repeatable executor");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    expanded_x_cache_.release();
    expanded_row_idx_cache_.release();
    expert_tokens_count_cache_.release();
    dynamic_quant_scale_cache_.release();
    x_cache_.release();
    expert_idx_cache_.release();
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor expanded_x, Tensor expanded_row_idx,
                  Tensor expert_tokens_count, Tensor dynamic_quant_scale,
                  const Tensor x, const Tensor expert_idx) const override {
    auto* t_expanded_x = expanded_x_cache_.get(expanded_x.data());
    auto* t_expanded_row_idx =
        expanded_row_idx_cache_.get(expanded_row_idx.data());
    auto* t_expert_tokens_count =
        expert_tokens_count_cache_.get(expert_tokens_count.data());
    auto* t_dynamic_quant_scale =
        dynamic_quant_scale_cache_.get(dynamic_quant_scale.data());
    auto* t_x = x_cache_.get(const_cast<void*>(x.data()));
    auto* t_expert_idx =
        expert_idx_cache_.get(const_cast<void*>(expert_idx.data()));
    ascend::CheckAcl(aclSetInputTensorAddr(
                         executor_, 0, t_x, const_cast<void*>(x.data())),
                     "DeepseekV4MoeInitRoutingQuant x address update");
    ascend::CheckAcl(aclSetInputTensorAddr(
                         executor_, 1, t_expert_idx,
                         const_cast<void*>(expert_idx.data())),
                     "DeepseekV4MoeInitRoutingQuant ids address update");
    ascend::CheckAcl(aclSetOutputTensorAddr(
                         executor_, 0, t_expanded_x, expanded_x.data()),
                     "DeepseekV4MoeInitRoutingQuant x output address update");
    ascend::CheckAcl(aclSetOutputTensorAddr(
                         executor_, 1, t_expanded_row_idx,
                         expanded_row_idx.data()),
                     "DeepseekV4MoeInitRoutingQuant row address update");
    ascend::CheckAcl(aclSetOutputTensorAddr(
                         executor_, 2, t_expert_tokens_count,
                         expert_tokens_count.data()),
                     "DeepseekV4MoeInitRoutingQuant count address update");
    ascend::CheckAcl(aclSetOutputTensorAddr(
                         executor_, 4, t_dynamic_quant_scale,
                         dynamic_quant_scale.data()),
                     "DeepseekV4MoeInitRoutingQuant scale address update");
    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size_, "deepseek_v4_moe_init_routing_quant");
    ascend::CheckAcl(aclnnMoeInitRoutingQuantV2(
                         arena.buf, workspace_size_, executor_, stream),
                     "DeepseekV4MoeInitRoutingQuant execution");
  }

 private:
  mutable ascend::AclTensorCache expanded_x_cache_;
  mutable ascend::AclTensorCache expanded_row_idx_cache_;
  mutable ascend::AclTensorCache expert_tokens_count_cache_;
  mutable ascend::AclTensorCache dynamic_quant_scale_cache_;
  mutable ascend::AclTensorCache x_cache_;
  mutable ascend::AclTensorCache expert_idx_cache_;
  aclTensor* t_expanded_x_{nullptr};
  aclTensor* t_expanded_row_idx_{nullptr};
  aclTensor* t_expert_tokens_count_{nullptr};
  aclTensor* t_dynamic_quant_scale_{nullptr};
  aclTensor* t_x_{nullptr};
  aclTensor* t_expert_idx_{nullptr};
  mutable aclOpExecutor* executor_{nullptr};
  uint64_t workspace_size_{0};
};

}  // namespace infini::ops

#endif
