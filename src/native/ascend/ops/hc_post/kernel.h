#ifndef INFINI_OPS_ASCEND_HC_POST_KERNEL_H_
#define INFINI_OPS_ASCEND_HC_POST_KERNEL_H_

#include <stdexcept>

#include "base/hc_post.h"
#include "operator.h"

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_hc_post.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#endif

namespace infini::ops {

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
template <>
class Operator<HcPost, Device::Type::kAscend, 0> : public HcPost {
 public:
  static bool supports(Tensor out, const Tensor x, const Tensor residual,
                       const Tensor post, const Tensor comb) {
    if (x.device().type() != Device::Type::kAscend || x.ndim() != 3 ||
        residual.ndim() != 4 || post.ndim() != 3 || comb.ndim() != 4 ||
        x.dtype() != DataType::kBFloat16 ||
        residual.dtype() != DataType::kBFloat16 ||
        out.dtype() != DataType::kBFloat16 ||
        post.dtype() != DataType::kFloat32 ||
        comb.dtype() != DataType::kFloat32 ||
        residual.device() != x.device() || out.device() != x.device() ||
        post.device() != x.device() || comb.device() != x.device() ||
        !x.IsContiguous() || !residual.IsContiguous() ||
        !out.IsContiguous() || !post.IsContiguous() ||
        !comb.IsContiguous()) {
      return false;
    }

    const auto batch = x.shape()[0];
    const auto sequence = x.shape()[1];
    const auto d = x.shape()[2];
    return residual.shape() == Tensor::Shape{batch, sequence, 4, d} &&
           out.shape() == residual.shape() &&
           post.shape() == Tensor::Shape{batch, sequence, 4} &&
           comb.shape() == Tensor::Shape{batch, sequence, 4, 4};
  }

  Operator(Tensor out, const Tensor x, const Tensor residual,
           const Tensor post, const Tensor comb)
      : HcPost(out, x, residual, post, comb),
        out_cache_(out),
        x_cache_(x),
        residual_cache_(residual),
        post_cache_(post),
        comb_cache_(comb),
        t_out_(out_cache_.get(out.data())),
        t_x_(x_cache_.get(const_cast<void*>(x.data()))),
        t_residual_(residual_cache_.get(const_cast<void*>(residual.data()))),
        t_post_(post_cache_.get(const_cast<void*>(post.data()))),
        t_comb_(comb_cache_.get(const_cast<void*>(comb.data()))) {
    if (!supports(out, x, residual, post, comb)) {
      throw std::invalid_argument("unsupported Ascend HcPost operands");
    }
    ascend::CheckAcl(aclnnHcPostGetWorkspaceSize(
                         t_x_, t_residual_, t_post_, t_comb_, t_out_,
                         &workspace_size_, &executor_),
                     "HcPost workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                     "HcPost repeatable executor");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    out_cache_.release();
    x_cache_.release();
    residual_cache_.release();
    post_cache_.release();
    comb_cache_.release();
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor out, const Tensor x, const Tensor residual,
                  const Tensor post, const Tensor comb) const override {
    if (!supports(out, x, residual, post, comb)) {
      throw std::invalid_argument("unsupported Ascend HcPost operands");
    }

    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, 0, t_x_, const_cast<void*>(x.data())),
                     "HcPost x address");
    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, 1, t_residual_,
                         const_cast<void*>(residual.data())),
                     "HcPost residual address");
    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, 2, t_post_,
                         const_cast<void*>(post.data())),
                     "HcPost post address");
    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, 3, t_comb_,
                         const_cast<void*>(comb.data())),
                     "HcPost comb address");
    ascend::CheckAcl(
        AclSetOutputTensorAddr(executor_, 0, t_out_, out.data()),
        "HcPost output address");

    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size_,
                                                     "hc_post");
    ascend::CheckAcl(
        aclnnHcPost(arena.buf, workspace_size_, executor_, stream),
        "HcPost execution");
  }

 private:
  mutable ascend::AclTensorCache out_cache_;
  mutable ascend::AclTensorCache x_cache_;
  mutable ascend::AclTensorCache residual_cache_;
  mutable ascend::AclTensorCache post_cache_;
  mutable ascend::AclTensorCache comb_cache_;
  aclTensor* t_out_{nullptr};
  aclTensor* t_x_{nullptr};
  aclTensor* t_residual_{nullptr};
  aclTensor* t_post_{nullptr};
  aclTensor* t_comb_{nullptr};
  mutable aclOpExecutor* executor_{nullptr};
  uint64_t workspace_size_{0};
};
#else
template <>
class Operator<HcPost, Device::Type::kAscend, 0> : public HcPost {
 public:
  template <typename... Args>
  static bool supports(const Args&...) {
    return false;
  }

  Operator(Tensor out, const Tensor x, const Tensor residual,
           const Tensor post, const Tensor comb)
      : HcPost(out, x, residual, post, comb) {
    throw std::runtime_error(
        "HcPost requires INFINI_ASCEND_CUSTOM_OPAPI_ROOT at build time");
  }

  void operator()(Tensor, const Tensor, const Tensor, const Tensor,
                  const Tensor) const override {
    throw std::runtime_error("HcPost Ascend provider is unavailable");
  }
};
#endif

}  // namespace infini::ops

#endif
