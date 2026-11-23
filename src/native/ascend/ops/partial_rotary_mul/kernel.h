#ifndef INFINI_OPS_ASCEND_PARTIAL_ROTARY_MUL_KERNEL_H_
#define INFINI_OPS_ASCEND_PARTIAL_ROTARY_MUL_KERNEL_H_

#include <array>
#include <stdexcept>

#include "base/partial_rotary_mul.h"
#include "operator.h"

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_inplace_partial_rotary_mul.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#endif

namespace infini::ops {

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
template <>
class Operator<PartialRotaryMul, Device::Type::kAscend, 0>
    : public PartialRotaryMul {
 public:
  static bool supports(Tensor x, const Tensor cos, const Tensor sin,
                       int64_t mode, int64_t slice_begin,
                       int64_t slice_end) {
    return x.device().type() == Device::Type::kAscend && x.ndim() == 4 &&
           x.device() == cos.device() && x.device() == sin.device() &&
           cos.shape() == sin.shape() && cos.dtype() == sin.dtype() &&
           mode >= 0 && mode <= 3 && slice_begin >= 0 &&
           slice_end > slice_begin &&
           slice_end <= static_cast<int64_t>(x.shape().back()) &&
           x.IsContiguous() && cos.IsContiguous() && sin.IsContiguous();
  }

  Operator(Tensor x, const Tensor cos, const Tensor sin, int64_t mode,
           int64_t slice_begin, int64_t slice_end)
      : PartialRotaryMul(x, cos, sin, mode, slice_begin, slice_end),
        x_cache_(x),
        cos_cache_(cos),
        sin_cache_(sin),
        slice_{slice_begin, slice_end},
        mode_(mode),
        t_x_(x_cache_.get(x.data())),
        t_cos_(cos_cache_.get(const_cast<void*>(cos.data()))),
        t_sin_(sin_cache_.get(const_cast<void*>(sin.data()))) {
    if (!supports(x, cos, sin, mode, slice_begin, slice_end)) {
      throw std::invalid_argument(
          "unsupported Ascend PartialRotaryMul operands");
    }
    slice_array_ = aclCreateIntArray(slice_.data(), slice_.size());
    if (slice_array_ == nullptr) {
      throw std::runtime_error("PartialRotaryMul slice creation failed");
    }
    ascend::CheckAcl(aclnnInplacePartialRotaryMulGetWorkspaceSize(
                         t_x_, t_cos_, t_sin_, mode, slice_array_,
                         &workspace_size_, &executor_),
                     "PartialRotaryMul workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                     "PartialRotaryMul repeatable executor");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    x_cache_.release();
    cos_cache_.release();
    sin_cache_.release();
    // The repeatable executor retains slice_array_; keep both alive together.
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor x, const Tensor cos, const Tensor sin, int64_t mode,
                  int64_t slice_begin, int64_t slice_end) const override {
    if (!supports(x, cos, sin, mode, slice_begin, slice_end) ||
        mode != mode_ || slice_begin != slice_[0] || slice_end != slice_[1]) {
      throw std::invalid_argument(
          "unsupported Ascend PartialRotaryMul operands");
    }
    ascend::CheckAcl(AclSetInputTensorAddr(executor_, 0, t_x_, x.data()),
                     "PartialRotaryMul x address");
    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, 1, t_cos_, const_cast<void*>(cos.data())),
                     "PartialRotaryMul cos address");
    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, 2, t_sin_, const_cast<void*>(sin.data())),
                     "PartialRotaryMul sin address");
    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size_, "partial_rotary_mul");
    ascend::CheckAcl(aclnnInplacePartialRotaryMul(
                         arena.buf, workspace_size_, executor_, stream),
                     "PartialRotaryMul execution");
  }

 private:
  mutable ascend::AclTensorCache x_cache_;
  mutable ascend::AclTensorCache cos_cache_;
  mutable ascend::AclTensorCache sin_cache_;
  std::array<int64_t, 2> slice_;
  int64_t mode_{0};
  aclTensor* t_x_{nullptr};
  aclTensor* t_cos_{nullptr};
  aclTensor* t_sin_{nullptr};
  aclIntArray* slice_array_{nullptr};
  mutable aclOpExecutor* executor_{nullptr};
  uint64_t workspace_size_{0};
};
#else
template <>
class Operator<PartialRotaryMul, Device::Type::kAscend, 0>
    : public PartialRotaryMul {
 public:
  template <typename... Args>
  static bool supports(const Args&...) {
    return false;
  }
  Operator(Tensor x, const Tensor cos, const Tensor sin, int64_t mode,
           int64_t slice_begin, int64_t slice_end)
      : PartialRotaryMul(x, cos, sin, mode, slice_begin, slice_end) {
    throw std::runtime_error(
        "PartialRotaryMul requires INFINI_ASCEND_CUSTOM_OPAPI_ROOT");
  }
  void operator()(Tensor, const Tensor, const Tensor, int64_t, int64_t,
                  int64_t) const override {
    throw std::runtime_error(
        "PartialRotaryMul Ascend provider is unavailable");
  }
};
#endif

}  // namespace infini::ops

#endif
