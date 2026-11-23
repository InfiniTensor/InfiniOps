#ifndef INFINI_OPS_ASCEND_DEEPSEEK_V4_MOE_GROUPED_MATMUL_KERNEL_H_
#define INFINI_OPS_ASCEND_DEEPSEEK_V4_MOE_GROUPED_MATMUL_KERNEL_H_

#include <optional>
#include <stdexcept>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_grouped_matmul_v5.h"
#include "base/deepseek_v4_moe_grouped_matmul.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

namespace {
std::vector<int64_t> nz_storage_shape(const Tensor& weight) {
  return std::vector<int64_t>(weight.shape().begin(), weight.shape().end());
}

Tensor nd_weight_view(const Tensor& weight) {
  return Tensor{const_cast<void*>(weight.data()),
                {weight.shape()[0],
                 static_cast<size_t>(weight.shape()[2]) * 16,
                 static_cast<size_t>(weight.shape()[1]) * 32},
                weight.dtype()};
}

aclFormat weight_storage_format(const Tensor& weight) {
  return weight.ndim() == 5 ? ACL_FORMAT_FRACTAL_NZ : ACL_FORMAT_ND;
}
}

template <>
class Operator<DeepseekV4MoeGroupedMatmul, Device::Type::kAscend, 0>
    : public DeepseekV4MoeGroupedMatmul {
 public:
  static bool supports(Tensor out, const Tensor x, const Tensor weight,
                       const std::optional<Tensor> weight_scale,
                       const std::optional<Tensor> per_token_scale,
                       const Tensor group_list, int64_t group_list_type) {
    const bool quantized_output = weight_scale.has_value();
    const bool scale_pair =
        quantized_output == per_token_scale.has_value();
    const bool dtype_ok = quantized_output
        ? (weight_scale->dtype() == DataType::kBFloat16 &&
           per_token_scale->dtype() == DataType::kFloat32 &&
           out.dtype() == DataType::kBFloat16)
        : out.dtype() == DataType::kInt32;
    return x.device().type() == Device::Type::kAscend &&
           x.dtype() == DataType::kInt8 && x.ndim() == 2 &&
           x.IsContiguous() && weight.device() == x.device() &&
           weight.dtype() == DataType::kInt8 &&
           (weight.ndim() == 3 || (weight.ndim() == 5 && quantized_output)) &&
           weight.IsContiguous() && out.device() == x.device() &&
           out.ndim() == 2 && out.IsContiguous() && scale_pair && dtype_ok &&
           (!weight_scale ||
            (weight_scale->device() == x.device() &&
             weight_scale->IsContiguous())) &&
           (!per_token_scale ||
            (per_token_scale->device() == x.device() &&
             per_token_scale->IsContiguous())) &&
           group_list.device() == x.device() &&
           group_list.dtype() == DataType::kInt64 &&
           group_list.IsContiguous() &&
           (group_list_type == 0 || group_list_type == 1);
  }

  Operator(Tensor out, const Tensor x, const Tensor weight,
           const std::optional<Tensor> weight_scale,
           const std::optional<Tensor> per_token_scale,
           const Tensor group_list, int64_t group_list_type)
      : DeepseekV4MoeGroupedMatmul(out, x, weight, weight_scale,
                                   per_token_scale, group_list,
                                   group_list_type),
        out_cache_(out),
        x_cache_(x),
        weight_cache_(weight.ndim() == 5 ? nd_weight_view(weight) : weight,
                      false,
                      weight.ndim() == 5 ? nz_storage_shape(weight)
                                         : std::vector<int64_t>{},
                      weight_storage_format(weight)),
        weight_scale_cache_(
            weight_scale ? ascend::AclTensorCache(*weight_scale)
                         : ascend::AclTensorCache()),
        per_token_scale_cache_(
            per_token_scale ? ascend::AclTensorCache(*per_token_scale)
                            : ascend::AclTensorCache()),
        group_list_cache_(group_list),
        has_scale_(static_cast<bool>(weight_scale)),
        group_list_type_(group_list_type),
        split_item_(has_scale_ ? 2 : 3) {
    if (!supports(out, x, weight, weight_scale, per_token_scale, group_list,
                  group_list_type)) {
      throw std::invalid_argument(
          "unsupported Ascend DeepseekV4MoeGroupedMatmul operands");
    }
    t_out_ = out_cache_.get(out.data());
    t_x_ = x_cache_.get(const_cast<void*>(x.data()));
    t_weight_ = weight_cache_.get(const_cast<void*>(weight.data()));
    t_group_list_ =
        group_list_cache_.get(const_cast<void*>(group_list.data()));
    x_list_ = make_list(t_x_);
    weight_list_ = make_list(t_weight_);
    out_list_ = make_list(t_out_);
    if (has_scale_) {
      t_weight_scale_ = weight_scale_cache_.get(
          const_cast<void*>(weight_scale->data()));
      t_per_token_scale_ = per_token_scale_cache_.get(
          const_cast<void*>(per_token_scale->data()));
      weight_scale_list_ = make_list(t_weight_scale_);
      per_token_scale_list_ = make_list(t_per_token_scale_);
    }
    ascend::CheckAcl(aclnnGroupedMatmulV5GetWorkspaceSize(
                         x_list_, weight_list_, nullptr, weight_scale_list_,
                         nullptr, nullptr, nullptr, per_token_scale_list_,
                         t_group_list_, nullptr, nullptr, nullptr, split_item_,
                         0, group_list_type_, 0, nullptr, out_list_, nullptr,
                         nullptr, &workspace_size_, &executor_),
                     "DeepseekV4MoeGroupedMatmul workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                     "DeepseekV4MoeGroupedMatmul repeatable executor");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    out_cache_.release();
    x_cache_.release();
    weight_cache_.release();
    group_list_cache_.release();
    if (has_scale_) {
      weight_scale_cache_.release();
      per_token_scale_cache_.release();
    }
    if (x_list_) aclDestroyTensorList(x_list_);
    if (weight_list_) aclDestroyTensorList(weight_list_);
    if (weight_scale_list_) aclDestroyTensorList(weight_scale_list_);
    if (per_token_scale_list_) aclDestroyTensorList(per_token_scale_list_);
    if (out_list_) aclDestroyTensorList(out_list_);
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor out, const Tensor x, const Tensor weight,
                  const std::optional<Tensor> weight_scale,
           const std::optional<Tensor> per_token_scale,
                  const Tensor group_list, int64_t) const override {
    out_cache_.get(out.data());
    x_cache_.get(const_cast<void*>(x.data()));
    weight_cache_.get(const_cast<void*>(weight.data()));
    auto* t_group_list =
        group_list_cache_.get(const_cast<void*>(group_list.data()));
    ascend::CheckAcl(aclSetInputTensorAddr(
                         executor_, 8, t_group_list,
                         const_cast<void*>(group_list.data())),
                     "DeepseekV4MoeGroupedMatmul group address update");
    if (has_scale_) {
      weight_scale_cache_.get(const_cast<void*>(weight_scale->data()));
      per_token_scale_cache_.get(
          const_cast<void*>(per_token_scale->data()));
    }
    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size_, "deepseek_v4_moe_grouped_matmul");
    ascend::CheckAcl(aclnnGroupedMatmulV5(
                         arena.buf, workspace_size_, executor_, stream),
                     "DeepseekV4MoeGroupedMatmul execution");
  }

 private:
  static aclTensorList* make_list(aclTensor* tensor) {
    const aclTensor* tensors[] = {tensor};
    auto* list = aclCreateTensorList(tensors, 1);
    if (!list) throw std::runtime_error("ACL tensor-list creation failed");
    return list;
  }

  mutable ascend::AclTensorCache out_cache_;
  mutable ascend::AclTensorCache x_cache_;
  mutable ascend::AclTensorCache weight_cache_;
  mutable ascend::AclTensorCache weight_scale_cache_;
  mutable ascend::AclTensorCache per_token_scale_cache_;
  mutable ascend::AclTensorCache group_list_cache_;
  bool has_scale_{false};
  int64_t group_list_type_{0};
  int64_t split_item_{3};
  aclTensor* t_out_{nullptr};
  aclTensor* t_x_{nullptr};
  aclTensor* t_weight_{nullptr};
  aclTensor* t_weight_scale_{nullptr};
  aclTensor* t_per_token_scale_{nullptr};
  aclTensor* t_group_list_{nullptr};
  aclTensorList* x_list_{nullptr};
  aclTensorList* weight_list_{nullptr};
  aclTensorList* weight_scale_list_{nullptr};
  aclTensorList* per_token_scale_list_{nullptr};
  aclTensorList* out_list_{nullptr};
  mutable aclOpExecutor* executor_{nullptr};
  uint64_t workspace_size_{0};
};

}  // namespace infini::ops

#endif
