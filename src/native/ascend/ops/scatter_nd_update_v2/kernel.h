#ifndef INFINI_OPS_ASCEND_SCATTER_ND_UPDATE_V2_KERNEL_H_
#define INFINI_OPS_ASCEND_SCATTER_ND_UPDATE_V2_KERNEL_H_

#include <stdexcept>
#include <vector>

#include "base/scatter_nd_update_v2.h"
#include "operator.h"

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_scatter_nd_update_v2.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#endif

namespace infini::ops {

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
template <>
class Operator<ScatterNdUpdateV2, Device::Type::kAscend, 0>
    : public ScatterNdUpdateV2 {
 public:
  static bool supports(Tensor var, const Tensor indices,
                       const Tensor updates) {
    const auto index_type = indices.dtype();
    return var.device().type() == Device::Type::kAscend &&
           indices.device() == var.device() &&
           updates.device() == var.device() &&
           updates.dtype() == var.dtype() &&
           (index_type == DataType::kInt32 ||
            index_type == DataType::kInt64) &&
           indices.ndim() >= 1 && indices.shape().back() <= var.ndim() &&
           var.IsContiguous() && indices.IsContiguous() &&
           updates.IsContiguous();
  }

  Operator(Tensor var, const Tensor indices, const Tensor updates)
      : ScatterNdUpdateV2(var, indices, updates),
        var_cache_(var),
        indices_cache_(indices),
        updates_cache_(updates),
        strides_(var.strides().begin(), var.strides().end()),
        t_var_(var_cache_.get(var.data())),
        t_indices_(indices_cache_.get(const_cast<void*>(indices.data()))),
        t_updates_(updates_cache_.get(const_cast<void*>(updates.data()))) {
    if (!supports(var, indices, updates)) {
      throw std::invalid_argument(
          "unsupported Ascend ScatterNdUpdateV2 operands");
    }
    strides_array_ = aclCreateIntArray(strides_.data(), strides_.size());
    if (strides_array_ == nullptr) {
      throw std::runtime_error("ScatterNdUpdateV2 strides creation failed");
    }
    ascend::CheckAcl(aclnnScatterNdUpdateV2GetWorkspaceSize(
                         t_var_, t_indices_, t_updates_, strides_array_,
                         &workspace_size_, &executor_),
                     "ScatterNdUpdateV2 workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                     "ScatterNdUpdateV2 repeatable executor");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    var_cache_.release();
    indices_cache_.release();
    updates_cache_.release();
    // The repeatable executor retains strides_array_.
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor var, const Tensor indices,
                  const Tensor updates) const override {
    if (!supports(var, indices, updates)) {
      throw std::invalid_argument(
          "unsupported Ascend ScatterNdUpdateV2 operands");
    }
    ascend::CheckAcl(AclSetInputTensorAddr(executor_, 0, t_var_, var.data()),
                     "ScatterNdUpdateV2 var address");
    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, 1, t_indices_,
                         const_cast<void*>(indices.data())),
                     "ScatterNdUpdateV2 indices address");
    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, 2, t_updates_,
                         const_cast<void*>(updates.data())),
                     "ScatterNdUpdateV2 updates address");
    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size_, "scatter_nd_update_v2");
    ascend::CheckAcl(aclnnScatterNdUpdateV2(
                         arena.buf, workspace_size_, executor_, stream),
                     "ScatterNdUpdateV2 execution");
  }

 private:
  mutable ascend::AclTensorCache var_cache_;
  mutable ascend::AclTensorCache indices_cache_;
  mutable ascend::AclTensorCache updates_cache_;
  std::vector<int64_t> strides_;
  aclTensor* t_var_{nullptr};
  aclTensor* t_indices_{nullptr};
  aclTensor* t_updates_{nullptr};
  aclIntArray* strides_array_{nullptr};
  mutable aclOpExecutor* executor_{nullptr};
  uint64_t workspace_size_{0};
};
#else
template <>
class Operator<ScatterNdUpdateV2, Device::Type::kAscend, 0>
    : public ScatterNdUpdateV2 {
 public:
  template <typename... Args>
  static bool supports(const Args&...) {
    return false;
  }

  Operator(Tensor var, const Tensor indices, const Tensor updates)
      : ScatterNdUpdateV2(var, indices, updates) {
    throw std::runtime_error(
        "ScatterNdUpdateV2 requires INFINI_ASCEND_CUSTOM_OPAPI_ROOT");
  }

  void operator()(Tensor, const Tensor, const Tensor) const override {
    throw std::runtime_error(
        "ScatterNdUpdateV2 Ascend provider is unavailable");
  }
};
#endif

}  // namespace infini::ops

#endif
