#ifndef INFINI_OPS_ASCEND_HC_PRE_KERNEL_H_
#define INFINI_OPS_ASCEND_HC_PRE_KERNEL_H_

#include <stdexcept>

#include "base/hc_pre.h"
#include "operator.h"

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_hc_pre.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#endif

namespace infini::ops {

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
template <>
class Operator<HcPre, Device::Type::kAscend, 0> : public HcPre {
 public:
  static bool supports(Tensor y, Tensor post, Tensor comb, const Tensor x,
                       const Tensor hc_fn, const Tensor hc_scale,
                       const Tensor hc_base, int64_t hc_mult,
                       int64_t hc_sinkhorn_iters, double norm_eps,
                       double hc_eps) {
    if (x.device().type() != Device::Type::kAscend ||
        x.dtype() != DataType::kBFloat16 ||
        y.dtype() != DataType::kBFloat16 ||
        post.dtype() != DataType::kFloat32 ||
        comb.dtype() != DataType::kFloat32 || hc_fn.dtype() != DataType::kFloat32 ||
        hc_scale.dtype() != DataType::kFloat32 ||
        hc_base.dtype() != DataType::kFloat32 ||
        y.device() != x.device() || post.device() != x.device() ||
        comb.device() != x.device() || hc_fn.device() != x.device() ||
        hc_scale.device() != x.device() || hc_base.device() != x.device() ||
        (x.ndim() != 3 && x.ndim() != 4) || hc_mult != 4 ||
        hc_sinkhorn_iters <= 0 || norm_eps < 0.0 || hc_eps < 0.0 ||
        !x.IsContiguous() || !y.IsContiguous() || !post.IsContiguous() ||
        !comb.IsContiguous() || !hc_fn.IsContiguous() ||
        !hc_scale.IsContiguous() || !hc_base.IsContiguous()) {
      return false;
    }

    const auto hc_axis = x.ndim() == 4 ? 2 : 1;
    const auto d_axis = x.ndim() == 4 ? 3 : 2;
    const auto d = x.shape()[d_axis];
    if (x.shape()[hc_axis] != 4 || (d != 4096 && d != 7168) ||
        hc_fn.shape() != Tensor::Shape{24, 4 * d} ||
        hc_scale.shape() != Tensor::Shape{3} ||
        hc_base.shape() != Tensor::Shape{24}) {
      return false;
    }

    if (x.ndim() == 3) {
      const auto tokens = x.shape()[0];
      return y.shape() == Tensor::Shape{tokens, d} &&
             post.shape() == Tensor::Shape{tokens, 4} &&
             comb.shape() == Tensor::Shape{tokens, 4, 4};
    }

    const auto batch = x.shape()[0];
    const auto sequence = x.shape()[1];
    return y.shape() == Tensor::Shape{batch, sequence, d} &&
           post.shape() == Tensor::Shape{batch, sequence, 4} &&
           comb.shape() == Tensor::Shape{batch, sequence, 4, 4};
  }

  Operator(Tensor y, Tensor post, Tensor comb, const Tensor x,
           const Tensor hc_fn, const Tensor hc_scale, const Tensor hc_base,
           int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps,
           double hc_eps)
      : HcPre(y, post, comb, x, hc_fn, hc_scale, hc_base, hc_mult,
              hc_sinkhorn_iters, norm_eps, hc_eps),
        y_cache_(y),
        post_cache_(post),
        comb_cache_(comb),
        x_cache_(x),
        hc_fn_cache_(hc_fn),
        hc_scale_cache_(hc_scale),
        hc_base_cache_(hc_base),
        t_y_(y_cache_.get(y.data())),
        t_post_(post_cache_.get(post.data())),
        t_comb_(comb_cache_.get(comb.data())),
        t_x_(x_cache_.get(const_cast<void*>(x.data()))),
        t_hc_fn_(hc_fn_cache_.get(const_cast<void*>(hc_fn.data()))),
        t_hc_scale_(hc_scale_cache_.get(const_cast<void*>(hc_scale.data()))),
        t_hc_base_(hc_base_cache_.get(const_cast<void*>(hc_base.data()))) {
    if (!supports(y, post, comb, x, hc_fn, hc_scale, hc_base, hc_mult,
                  hc_sinkhorn_iters, norm_eps, hc_eps)) {
      throw std::invalid_argument("unsupported Ascend HcPre operands");
    }
    ascend::CheckAcl(
        aclnnHcPreGetWorkspaceSize(
            t_x_, t_hc_fn_, t_hc_scale_, t_hc_base_, hc_mult,
            hc_sinkhorn_iters, hc_eps, norm_eps, t_y_, t_post_, t_comb_,
            &workspace_size_, &executor_),
        "HcPre workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                     "HcPre repeatable executor");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    y_cache_.release();
    post_cache_.release();
    comb_cache_.release();
    x_cache_.release();
    hc_fn_cache_.release();
    hc_scale_cache_.release();
    hc_base_cache_.release();
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor y, Tensor post, Tensor comb, const Tensor x,
                  const Tensor hc_fn, const Tensor hc_scale,
                  const Tensor hc_base, int64_t hc_mult,
                  int64_t hc_sinkhorn_iters, double norm_eps,
                  double hc_eps) const override {
    if (!supports(y, post, comb, x, hc_fn, hc_scale, hc_base, hc_mult,
                  hc_sinkhorn_iters, norm_eps, hc_eps)) {
      throw std::invalid_argument("unsupported Ascend HcPre operands");
    }

    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, 0, t_x_, const_cast<void*>(x.data())),
                     "HcPre x address");
    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, 1, t_hc_fn_,
                         const_cast<void*>(hc_fn.data())),
                     "HcPre hc_fn address");
    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, 2, t_hc_scale_,
                         const_cast<void*>(hc_scale.data())),
                     "HcPre hc_scale address");
    ascend::CheckAcl(AclSetInputTensorAddr(
                         executor_, 3, t_hc_base_,
                         const_cast<void*>(hc_base.data())),
                     "HcPre hc_base address");
    ascend::CheckAcl(AclSetOutputTensorAddr(executor_, 0, t_y_, y.data()),
                     "HcPre y address");
    ascend::CheckAcl(
        AclSetOutputTensorAddr(executor_, 1, t_post_, post.data()),
        "HcPre post address");
    ascend::CheckAcl(
        AclSetOutputTensorAddr(executor_, 2, t_comb_, comb.data()),
        "HcPre comb address");

    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size_,
                                                     "hc_pre");
    ascend::CheckAcl(
        aclnnHcPre(arena.buf, workspace_size_, executor_, stream),
        "HcPre execution");
  }

 private:
  mutable ascend::AclTensorCache y_cache_;
  mutable ascend::AclTensorCache post_cache_;
  mutable ascend::AclTensorCache comb_cache_;
  mutable ascend::AclTensorCache x_cache_;
  mutable ascend::AclTensorCache hc_fn_cache_;
  mutable ascend::AclTensorCache hc_scale_cache_;
  mutable ascend::AclTensorCache hc_base_cache_;
  aclTensor* t_y_{nullptr};
  aclTensor* t_post_{nullptr};
  aclTensor* t_comb_{nullptr};
  aclTensor* t_x_{nullptr};
  aclTensor* t_hc_fn_{nullptr};
  aclTensor* t_hc_scale_{nullptr};
  aclTensor* t_hc_base_{nullptr};
  mutable aclOpExecutor* executor_{nullptr};
  uint64_t workspace_size_{0};
};
#else
template <>
class Operator<HcPre, Device::Type::kAscend, 0> : public HcPre {
 public:
  template <typename... Args>
  static bool supports(const Args&...) {
    return false;
  }

  Operator(Tensor y, Tensor post, Tensor comb, const Tensor x,
           const Tensor hc_fn, const Tensor hc_scale, const Tensor hc_base,
           int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps,
           double hc_eps)
      : HcPre(y, post, comb, x, hc_fn, hc_scale, hc_base, hc_mult,
              hc_sinkhorn_iters, norm_eps, hc_eps) {
    throw std::runtime_error(
        "HcPre requires INFINI_ASCEND_CUSTOM_OPAPI_ROOT at build time");
  }

  void operator()(Tensor, Tensor, Tensor, const Tensor, const Tensor,
                  const Tensor, const Tensor, int64_t, int64_t, double,
                  double) const override {
    throw std::runtime_error("HcPre Ascend provider is unavailable");
  }
};
#endif

}  // namespace infini::ops

#endif
