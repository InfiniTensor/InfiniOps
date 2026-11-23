#ifndef INFINI_OPS_ASCEND_VLLM_QUANT_LIGHTNING_INDEXER_KERNEL_H_
#define INFINI_OPS_ASCEND_VLLM_QUANT_LIGHTNING_INDEXER_KERNEL_H_

#include <stdexcept>
#include <string>

#include "base/vllm_quant_lightning_indexer.h"
#include "operator.h"

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_vllm_quant_lightning_indexer.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#endif

namespace infini::ops {

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER

template <>
class Operator<VllmQuantLightningIndexer, Device::Type::kAscend, 0>
    : public VllmQuantLightningIndexer {
 public:
  static bool supports(Tensor indices, Tensor values, const Tensor q,
                       const Tensor k, const Tensor w, const Tensor qs,
                       const Tensor ks, const Tensor ql, const Tensor kl,
                       const Tensor bt, const Tensor md, int64_t, int64_t,
                       std::string lq, std::string lk, int64_t sc, int64_t,
                       int64_t, int64_t, int64_t, bool rv) {
    return indices.device().type() == Device::Type::kAscend &&
           q.device() == indices.device() && k.device() == indices.device() &&
           indices.dtype() == DataType::kInt32 &&
           values.dtype() == DataType::kFloat32 &&
           q.dtype() == DataType::kInt8 && k.dtype() == DataType::kInt8 &&
           w.dtype() == DataType::kFloat16 &&
           qs.dtype() == DataType::kFloat16 &&
           ks.dtype() == DataType::kFloat16 &&
           ql.dtype() == DataType::kInt32 && kl.dtype() == DataType::kInt32 &&
           bt.dtype() == DataType::kInt32 && md.dtype() == DataType::kInt32 &&
           q.ndim() == 3 && k.ndim() == 4 && k.shape()[2] == 1 &&
           indices.shape() ==
               Tensor::Shape{q.shape()[0], 1, static_cast<uint64_t>(sc)} &&
           values.shape() == Tensor::Shape{0} && lq == "TND" &&
           lk == "PA_BSND" && indices.IsContiguous() &&
           values.IsContiguous() && q.IsContiguous() && k.IsContiguous() &&
           w.IsContiguous() && qs.IsContiguous() && ks.IsContiguous() &&
           ql.IsContiguous() && kl.IsContiguous() && bt.IsContiguous() &&
           md.IsContiguous() && !rv;
  }

  Operator(Tensor indices, Tensor values, const Tensor q, const Tensor k,
           const Tensor w, const Tensor qs, const Tensor ks, const Tensor ql,
           const Tensor kl, const Tensor bt, const Tensor md, int64_t qq,
           int64_t kq, std::string lq, std::string lk, int64_t sc, int64_t sm,
           int64_t pt, int64_t nt, int64_t cr, bool rv)
      : VllmQuantLightningIndexer(indices, values, q, k, w, qs, ks, ql, kl, bt,
                                  md, qq, kq, lq, lk, sc, sm, pt, nt, cr, rv),
        indices_cache_(indices),
        values_cache_(values),
        q_cache_(q),
        k_cache_(k),
        w_cache_(w),
        qs_cache_(qs),
        ks_cache_(ks),
        ql_cache_(ql),
        kl_cache_(kl),
        bt_cache_(bt),
        md_cache_(md),
        qq_(qq),
        kq_(kq),
        lq_(std::move(lq)),
        lk_(std::move(lk)),
        sc_(sc),
        sm_(sm),
        pt_(pt),
        nt_(nt),
        cr_(cr),
        rv_(rv),
        stride_(static_cast<int64_t>(k.strides()[0])),
        scale_stride_(static_cast<int64_t>(ks.strides()[0])) {
    if (!supports(indices, values, q, k, w, qs, ks, ql, kl, bt, md, qq_, kq_,
                  lq_, lk_, sc_, sm_, pt_, nt_, cr_, rv_)) {
      throw std::invalid_argument("unsupported Ascend QLI operands");
    }
  }

  bool capture_safe() const override { return false; }

  void operator()(Tensor indices, Tensor values, const Tensor q, const Tensor k,
                  const Tensor w, const Tensor qs, const Tensor ks,
                  const Tensor ql, const Tensor kl, const Tensor bt,
                  const Tensor md, int64_t, int64_t, std::string, std::string,
                  int64_t, int64_t, int64_t, int64_t, int64_t,
                  bool) const override {
    auto* t_indices = indices_cache_.get(indices.data());
    auto* t_values = values_cache_.get(values.data());
    auto* t_q = q_cache_.get(const_cast<void*>(q.data()));
    auto* t_k = k_cache_.get(const_cast<void*>(k.data()));
    auto* t_w = w_cache_.get(const_cast<void*>(w.data()));
    auto* t_qs = qs_cache_.get(const_cast<void*>(qs.data()));
    auto* t_ks = ks_cache_.get(const_cast<void*>(ks.data()));
    auto* t_ql = ql_cache_.get(const_cast<void*>(ql.data()));
    auto* t_kl = kl_cache_.get(const_cast<void*>(kl.data()));
    auto* t_bt = bt_cache_.get(const_cast<void*>(bt.data()));
    auto* t_md = md_cache_.get(const_cast<void*>(md.data()));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    ascend::CheckAcl(
        aclnnVllmQuantLightningIndexerGetWorkspaceSize(
            t_q, t_k, t_w, t_qs, t_ks, t_ql, t_kl, t_bt, t_md, qq_, kq_,
            const_cast<char*>(lq_.c_str()), const_cast<char*>(lk_.c_str()), sc_,
            sm_, pt_, nt_, cr_, rv_, stride_, scale_stride_, t_indices,
            t_values, &workspace_size, &executor),
        "QLI workspace query");

    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size,
                                                     "qli");
    ascend::CheckAcl(
        aclnnVllmQuantLightningIndexer(arena.buf, workspace_size, executor,
                                       stream),
        "QLI execution");
  }

 private:
  mutable ascend::AclTensorCache indices_cache_;
  mutable ascend::AclTensorCache values_cache_;
  mutable ascend::AclTensorCache q_cache_;
  mutable ascend::AclTensorCache k_cache_;
  mutable ascend::AclTensorCache w_cache_;
  mutable ascend::AclTensorCache qs_cache_;
  mutable ascend::AclTensorCache ks_cache_;
  mutable ascend::AclTensorCache ql_cache_;
  mutable ascend::AclTensorCache kl_cache_;
  mutable ascend::AclTensorCache bt_cache_;
  mutable ascend::AclTensorCache md_cache_;
  int64_t qq_;
  int64_t kq_;
  std::string lq_;
  std::string lk_;
  int64_t sc_;
  int64_t sm_;
  int64_t pt_;
  int64_t nt_;
  int64_t cr_;
  bool rv_;
  int64_t stride_;
  int64_t scale_stride_;
};

#else

template <>
class Operator<VllmQuantLightningIndexer, Device::Type::kAscend, 0>
    : public VllmQuantLightningIndexer {
 public:
  template <typename... Args>
  static bool supports(const Args&...) {
    return false;
  }

  Operator(Tensor indices, Tensor values, const Tensor q, const Tensor k,
           const Tensor w, const Tensor qs, const Tensor ks, const Tensor ql,
           const Tensor kl, const Tensor bt, const Tensor md, int64_t qq,
           int64_t kq, std::string lq, std::string lk, int64_t sc, int64_t sm,
           int64_t pt, int64_t nt, int64_t cr, bool rv)
      : VllmQuantLightningIndexer(indices, values, q, k, w, qs, ks, ql, kl, bt,
                                  md, qq, kq, lq, lk, sc, sm, pt, nt, cr, rv) {
    throw std::runtime_error("QLI requires custom transformer");
  }

  void operator()(Tensor, Tensor, const Tensor, const Tensor, const Tensor,
                  const Tensor, const Tensor, const Tensor, const Tensor,
                  const Tensor, const Tensor, int64_t, int64_t, std::string,
                  std::string, int64_t, int64_t, int64_t, int64_t, int64_t,
                  bool) const override {
    throw std::runtime_error("QLI unavailable");
  }
};

#endif

}  // namespace infini::ops

#endif
