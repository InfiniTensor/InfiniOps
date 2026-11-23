#ifndef INFINI_OPS_ASCEND_DEEPSEEK_V4_MOE_DEQUANT_SWIGLU_QUANT_KERNEL_H_
#define INFINI_OPS_ASCEND_DEEPSEEK_V4_MOE_DEQUANT_SWIGLU_QUANT_KERNEL_H_

#include <stdexcept>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_dequant_swiglu_quant.h"
#include "base/deepseek_v4_moe_dequant_swiglu_quant.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<DeepseekV4MoeDequantSwigluQuant, Device::Type::kAscend, 0>
    : public DeepseekV4MoeDequantSwigluQuant {
 public:
  static bool supports(Tensor out, Tensor out_scale, const Tensor x,
                       const Tensor weight_scale,
                       const Tensor activation_scale,
                       const Tensor group_index) {
    return x.device().type() == Device::Type::kAscend &&
           x.dtype() == DataType::kInt32 && x.ndim() == 2 &&
           x.IsContiguous() && weight_scale.device() == x.device() &&
           weight_scale.dtype() == DataType::kFloat32 &&
           weight_scale.IsContiguous() &&
           activation_scale.device() == x.device() &&
           activation_scale.dtype() == DataType::kFloat32 &&
           activation_scale.IsContiguous() &&
           group_index.device() == x.device() &&
           group_index.dtype() == DataType::kInt64 &&
           group_index.IsContiguous() && out.device() == x.device() &&
           out.dtype() == DataType::kInt8 && out.IsContiguous() &&
           out_scale.device() == x.device() &&
           out_scale.dtype() == DataType::kFloat32 &&
           out_scale.IsContiguous();
  }

  Operator(Tensor out, Tensor out_scale, const Tensor x,
           const Tensor weight_scale, const Tensor activation_scale,
           const Tensor group_index)
      : DeepseekV4MoeDequantSwigluQuant(
            out, out_scale, x, weight_scale, activation_scale, group_index),
        out_cache_(out),
        out_scale_cache_(out_scale),
        x_cache_(x),
        weight_scale_cache_(weight_scale),
        activation_scale_cache_(activation_scale),
        group_index_cache_(group_index) {
    if (!supports(out, out_scale, x, weight_scale, activation_scale,
                  group_index)) {
      throw std::invalid_argument(
          "unsupported Ascend DeepseekV4MoeDequantSwigluQuant operands");
    }
    t_out_ = out_cache_.get(out.data());
    t_out_scale_ = out_scale_cache_.get(out_scale.data());
    t_x_ = x_cache_.get(const_cast<void*>(x.data()));
    t_weight_scale_ =
        weight_scale_cache_.get(const_cast<void*>(weight_scale.data()));
    t_activation_scale_ = activation_scale_cache_.get(
        const_cast<void*>(activation_scale.data()));
    t_group_index_ =
        group_index_cache_.get(const_cast<void*>(group_index.data()));
    char* quant_mode = const_cast<char*>("dynamic");
    ascend::CheckAcl(aclnnDequantSwigluQuantGetWorkspaceSize(
                         t_x_, t_weight_scale_, t_activation_scale_, nullptr,
                         nullptr, nullptr, t_group_index_, true, quant_mode,
                         t_out_, t_out_scale_, &workspace_size_, &executor_),
                     "DeepseekV4MoeDequantSwigluQuant workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                     "DeepseekV4MoeDequantSwigluQuant repeatable executor");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    out_cache_.release();
    out_scale_cache_.release();
    x_cache_.release();
    weight_scale_cache_.release();
    activation_scale_cache_.release();
    group_index_cache_.release();
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor out, Tensor out_scale, const Tensor x,
                  const Tensor weight_scale,
                  const Tensor activation_scale,
                  const Tensor group_index) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto* t_out = out_cache_.get(out.data());
    auto* t_out_scale = out_scale_cache_.get(out_scale.data());
    auto* t_x = x_cache_.get(const_cast<void*>(x.data()));
    auto* t_weight_scale =
        weight_scale_cache_.get(const_cast<void*>(weight_scale.data()));
    auto* t_activation_scale = activation_scale_cache_.get(
        const_cast<void*>(activation_scale.data()));
    auto* t_group_index =
        group_index_cache_.get(const_cast<void*>(group_index.data()));
    ascend::CheckAcl(aclSetInputTensorAddr(
                         executor_, 0, t_x, const_cast<void*>(x.data())),
                     "DeepseekV4MoeDequantSwigluQuant x address update");
    ascend::CheckAcl(aclSetInputTensorAddr(
                         executor_, 1, t_weight_scale,
                         const_cast<void*>(weight_scale.data())),
                     "DeepseekV4MoeDequantSwigluQuant weight address update");
    ascend::CheckAcl(aclSetInputTensorAddr(
                         executor_, 2, t_activation_scale,
                         const_cast<void*>(activation_scale.data())),
                     "DeepseekV4MoeDequantSwigluQuant activation address update");
    ascend::CheckAcl(aclSetInputTensorAddr(
                         executor_, 6, t_group_index,
                         const_cast<void*>(group_index.data())),
                     "DeepseekV4MoeDequantSwigluQuant group address update");
    ascend::CheckAcl(aclSetOutputTensorAddr(executor_, 0, t_out, out.data()),
                     "DeepseekV4MoeDequantSwigluQuant output address update");
    ascend::CheckAcl(aclSetOutputTensorAddr(
                         executor_, 1, t_out_scale, out_scale.data()),
                     "DeepseekV4MoeDequantSwigluQuant scale address update");
    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size_, "deepseek_v4_moe_dequant_swiglu_quant");
    ascend::CheckAcl(aclnnDequantSwigluQuant(
                         arena.buf, workspace_size_, executor_, stream),
                     "DeepseekV4MoeDequantSwigluQuant execution");
  }

 private:
  mutable ascend::AclTensorCache out_cache_;
  mutable ascend::AclTensorCache out_scale_cache_;
  mutable ascend::AclTensorCache x_cache_;
  mutable ascend::AclTensorCache weight_scale_cache_;
  mutable ascend::AclTensorCache activation_scale_cache_;
  mutable ascend::AclTensorCache group_index_cache_;
  aclTensor* t_out_{nullptr};
  aclTensor* t_out_scale_{nullptr};
  aclTensor* t_x_{nullptr};
  aclTensor* t_weight_scale_{nullptr};
  aclTensor* t_activation_scale_{nullptr};
  aclTensor* t_group_index_{nullptr};
  mutable aclOpExecutor* executor_{nullptr};
  uint64_t workspace_size_{0};
};

}  // namespace infini::ops

#endif
