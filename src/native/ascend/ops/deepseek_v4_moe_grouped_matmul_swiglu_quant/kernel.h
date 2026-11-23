#ifndef INFINI_OPS_ASCEND_DEEPSEEK_V4_MOE_GROUPED_MATMUL_SWIGLU_QUANT_KERNEL_H_
#define INFINI_OPS_ASCEND_DEEPSEEK_V4_MOE_GROUPED_MATMUL_SWIGLU_QUANT_KERNEL_H_

#include <stdexcept>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "base/deepseek_v4_moe_grouped_matmul_swiglu_quant.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
extern "C" aclnnStatus
aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
    const aclTensor *offset, const aclTensor *weightScale,
    const aclTensor *xScale, const aclTensor *groupList, double limited,
    aclTensor *output, aclTensor *outputScale, aclTensor *outputOffset,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern "C" aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZ(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);
#endif

namespace infini::ops {

Tensor swiglu_nd_weight_view(const Tensor& weight) {
  return Tensor{const_cast<void*>(weight.data()),
                {weight.shape()[0],
                 static_cast<size_t>(weight.shape()[2]) * 16,
                 static_cast<size_t>(weight.shape()[1]) * 32},
                weight.dtype()};
}

template <>
class Operator<DeepseekV4MoeGroupedMatmulSwigluQuant, Device::Type::kAscend, 0>
    : public DeepseekV4MoeGroupedMatmulSwigluQuant {
 public:
  static bool supports(Tensor output, Tensor output_scale, const Tensor x,
                       const Tensor weight, const Tensor weight_scale,
                       const Tensor activation_scale,
                       const Tensor group_ends) {
    return x.device().type() == Device::Type::kAscend &&
           x.dtype() == DataType::kInt8 && x.ndim() == 2 &&
           x.IsContiguous() && weight.device() == x.device() &&
           weight.dtype() == DataType::kInt8 && weight.ndim() == 5 &&
           weight.IsContiguous() && weight_scale.device() == x.device() &&
           weight_scale.dtype() == DataType::kFloat32 &&
           weight_scale.ndim() == 2 && weight_scale.IsContiguous() &&
           activation_scale.device() == x.device() &&
           activation_scale.dtype() == DataType::kFloat32 &&
           activation_scale.ndim() == 1 &&
           activation_scale.IsContiguous() &&
           group_ends.device() == x.device() &&
           group_ends.dtype() == DataType::kInt64 &&
           group_ends.IsContiguous() &&
           output.device() == x.device() &&
           output.dtype() == DataType::kInt8 &&
           output.ndim() == 2 && output.IsContiguous() &&
           output_scale.device() == x.device() &&
           output_scale.dtype() == DataType::kFloat32 &&
           output_scale.ndim() == 1 && output_scale.IsContiguous();
  }

  Operator(Tensor output, Tensor output_scale, const Tensor x,
           const Tensor weight, const Tensor weight_scale,
           const Tensor activation_scale, const Tensor group_ends,
           double swiglu_limit)
      : DeepseekV4MoeGroupedMatmulSwigluQuant(
            output, output_scale, x, weight, weight_scale,
            activation_scale, group_ends, swiglu_limit),
        output_cache_(output), output_scale_cache_(output_scale),
        x_cache_(x),
        weight_cache_(swiglu_nd_weight_view(weight), false,
                      std::vector<int64_t>(weight.shape().begin(),
                                           weight.shape().end())),
        weight_scale_cache_(weight_scale),
        activation_scale_cache_(activation_scale),
        group_ends_cache_(group_ends), swiglu_limit_(swiglu_limit) {
#ifndef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
    throw std::invalid_argument(
        "DeepseekV4 fused SwiGlu matmul requires custom transformer operators");
#else
    if (!supports(output, output_scale, x, weight, weight_scale,
                  activation_scale, group_ends)) {
      throw std::invalid_argument(
          "unsupported Ascend DeepseekV4 fused SwiGlu matmul operands");
    }
    t_output_ = output_cache_.get(output.data());
    t_output_scale_ = output_scale_cache_.get(output_scale.data());
    t_x_ = x_cache_.get(const_cast<void *>(x.data()));
    t_weight_ = weight_cache_.get(const_cast<void *>(weight.data()));
    t_weight_scale_ = weight_scale_cache_.get(
        const_cast<void *>(weight_scale.data()));
    t_activation_scale_ = activation_scale_cache_.get(
        const_cast<void *>(activation_scale.data()));
    t_group_ends_ = group_ends_cache_.get(
        const_cast<void *>(group_ends.data()));
    ascend::CheckAcl(
        aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize(
            t_x_, t_weight_, nullptr, nullptr, t_weight_scale_,
            t_activation_scale_, t_group_ends_, swiglu_limit_,
            t_output_, t_output_scale_, nullptr, &workspace_size_,
            &executor_),
        "DeepseekV4 fused SwiGlu matmul workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                     "DeepseekV4 fused SwiGlu matmul repeatable executor");
#endif
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    output_cache_.release();
    output_scale_cache_.release();
    x_cache_.release();
    weight_cache_.release();
    weight_scale_cache_.release();
    activation_scale_cache_.release();
    group_ends_cache_.release();
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor output, Tensor output_scale, const Tensor x,
                  const Tensor weight, const Tensor weight_scale,
                  const Tensor activation_scale, const Tensor group_ends,
                  double) const override {
#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
    output_cache_.get(output.data());
    output_scale_cache_.get(output_scale.data());
    x_cache_.get(const_cast<void *>(x.data()));
    weight_cache_.get(const_cast<void *>(weight.data()));
    weight_scale_cache_.get(const_cast<void *>(weight_scale.data()));
    activation_scale_cache_.get(
        const_cast<void *>(activation_scale.data()));
    group_ends_cache_.get(const_cast<void *>(group_ends.data()));
    auto stream = static_cast<aclrtStream>(stream_);
    auto &arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size_,
        "deepseek_v4_moe_grouped_matmul_swiglu_quant");
    ascend::CheckAcl(
        aclnnGroupedMatmulSwigluQuantWeightNZ(
            arena.buf, workspace_size_, executor_, stream),
        "DeepseekV4 fused SwiGlu matmul execution");
#else
    throw std::runtime_error(
        "DeepseekV4 fused SwiGlu matmul provider is unavailable");
#endif
  }

 private:
  mutable ascend::AclTensorCache output_cache_;
  mutable ascend::AclTensorCache output_scale_cache_;
  mutable ascend::AclTensorCache x_cache_;
  mutable ascend::AclTensorCache weight_cache_;
  mutable ascend::AclTensorCache weight_scale_cache_;
  mutable ascend::AclTensorCache activation_scale_cache_;
  mutable ascend::AclTensorCache group_ends_cache_;
  double swiglu_limit_{0};
  aclTensor *t_output_{nullptr};
  aclTensor *t_output_scale_{nullptr};
  aclTensor *t_x_{nullptr};
  aclTensor *t_weight_{nullptr};
  aclTensor *t_weight_scale_{nullptr};
  aclTensor *t_activation_scale_{nullptr};
  aclTensor *t_group_ends_{nullptr};
  mutable aclOpExecutor *executor_{nullptr};
  uint64_t workspace_size_{0};
};

} // namespace infini::ops

#endif
