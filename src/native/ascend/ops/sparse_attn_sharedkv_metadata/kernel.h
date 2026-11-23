#ifndef INFINI_OPS_ASCEND_SPARSE_ATTN_SHAREDKV_METADATA_KERNEL_H_
#define INFINI_OPS_ASCEND_SPARSE_ATTN_SHAREDKV_METADATA_KERNEL_H_

#include <stdexcept>
#include <string>

#include "base/sparse_attn_sharedkv_metadata.h"
#include "operator.h"

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_sparse_attn_sharedkv_metadata.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#endif

namespace infini::ops {

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER

template <>
class Operator<SparseAttnSharedkvMetadata, Device::Type::kAscend, 0>
    : public SparseAttnSharedkvMetadata {
 public:
  static bool supports(
      Tensor out, const Tensor q, const Tensor ok, const Tensor ck,
      const Tensor sq, const Tensor sk, int64_t, int64_t, int64_t, int64_t,
      int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
      int64_t, int64_t, std::string lq, std::string lk, bool, bool) {
    return out.device().type() == Device::Type::kAscend &&
           q.device() == out.device() && out.dtype() == DataType::kInt32 &&
           out.shape() == Tensor::Shape{1024} &&
           q.dtype() == DataType::kInt32 && ok.dtype() == DataType::kInt32 &&
           ck.dtype() == DataType::kInt32 && sq.dtype() == DataType::kInt32 &&
           sk.dtype() == DataType::kInt32 && q.ndim() == 1 && ok.ndim() == 1 &&
           ck.ndim() == 1 && sq.ndim() == 1 && sk.ndim() == 1 && lq == "TND" &&
           lk == "PA_ND" && out.IsContiguous() && q.IsContiguous() &&
           ok.IsContiguous() && ck.IsContiguous() && sq.IsContiguous() &&
           sk.IsContiguous();
  }

  Operator(Tensor out, const Tensor q, const Tensor ok, const Tensor ck,
           const Tensor sq, const Tensor sk, int64_t nhq, int64_t nhk,
           int64_t hd, int64_t bs, int64_t msq, int64_t msk, int64_t ot,
           int64_t ct, int64_t cr, int64_t om, int64_t cm, int64_t wl,
           int64_t wr, std::string lq, std::string lk, bool ho, bool hc)
      : SparseAttnSharedkvMetadata(out, q, ok, ck, sq, sk, nhq, nhk, hd, bs,
                                   msq, msk, ot, ct, cr, om, cm, wl, wr, lq,
                                   lk, ho, hc),
        out_cache_(out),
        q_cache_(q),
        ok_cache_(ok),
        ck_cache_(ck),
        sq_cache_(sq),
        sk_cache_(sk),
        nhq_(nhq),
        nhk_(nhk),
        hd_(hd),
        bs_(bs),
        msq_(msq),
        msk_(msk),
        ot_(ot),
        ct_(ct),
        cr_(cr),
        om_(om),
        cm_(cm),
        wl_(wl),
        wr_(wr),
        lq_(std::move(lq)),
        lk_(std::move(lk)),
        ho_(ho),
        hc_(hc) {
    if (!supports(out, q, ok, ck, sq, sk, nhq, nhk, hd, bs, msq, msk, ot,
                  ct, cr, om, cm, wl, wr, lq_, lk_, ho_, hc_)) {
      throw std::invalid_argument("unsupported SparseAttn metadata operands");
    }
  }

  bool capture_safe() const override { return true; }

  void operator()(Tensor out, const Tensor q, const Tensor ok, const Tensor ck,
                  const Tensor sq, const Tensor sk, int64_t, int64_t, int64_t,
                  int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                  int64_t, int64_t, int64_t, int64_t, std::string,
                  std::string, bool, bool) const override {
    auto* t_out = out_cache_.get(out.data());
    auto* t_q = q_cache_.get(const_cast<void*>(q.data()));
    auto* t_ok = ok_cache_.get(const_cast<void*>(ok.data()));
    auto* t_ck = ck_cache_.get(const_cast<void*>(ck.data()));
    auto* t_sq = sq_cache_.get(const_cast<void*>(sq.data()));
    auto* t_sk = sk_cache_.get(const_cast<void*>(sk.data()));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    ascend::CheckAcl(
        aclnnSparseAttnSharedkvMetadataGetWorkspaceSize(
            t_q, t_ok, t_ck, t_sq, t_sk, nhq_, nhk_, hd_, bs_, msq_, msk_,
            ot_, ct_, cr_, om_, cm_, wl_, wr_,
            const_cast<char*>(lq_.c_str()), const_cast<char*>(lk_.c_str()), ho_,
            hc_, t_out, &workspace_size, &executor),
        "SparseAttn metadata workspace query");

    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size,
                                                     "sparse_metadata");
    ascend::CheckAcl(aclnnSparseAttnSharedkvMetadata(
                         arena.buf, workspace_size, executor, stream),
                     "Sparse metadata execution");
  }

 private:
  mutable ascend::AclTensorCache out_cache_;
  mutable ascend::AclTensorCache q_cache_;
  mutable ascend::AclTensorCache ok_cache_;
  mutable ascend::AclTensorCache ck_cache_;
  mutable ascend::AclTensorCache sq_cache_;
  mutable ascend::AclTensorCache sk_cache_;
  int64_t nhq_;
  int64_t nhk_;
  int64_t hd_;
  int64_t bs_;
  int64_t msq_;
  int64_t msk_;
  int64_t ot_;
  int64_t ct_;
  int64_t cr_;
  int64_t om_;
  int64_t cm_;
  int64_t wl_;
  int64_t wr_;
  std::string lq_;
  std::string lk_;
  bool ho_;
  bool hc_;
};

#else

template <>
class Operator<SparseAttnSharedkvMetadata, Device::Type::kAscend, 0>
    : public SparseAttnSharedkvMetadata {
 public:
  template <typename... Args>
  static bool supports(const Args&...) {
    return false;
  }

  Operator(Tensor out, const Tensor q, const Tensor ok, const Tensor ck,
           const Tensor sq, const Tensor sk, int64_t nhq, int64_t nhk,
           int64_t hd, int64_t bs, int64_t msq, int64_t msk, int64_t ot,
           int64_t ct, int64_t cr, int64_t om, int64_t cm, int64_t wl,
           int64_t wr, std::string lq, std::string lk, bool ho, bool hc)
      : SparseAttnSharedkvMetadata(out, q, ok, ck, sq, sk, nhq, nhk, hd, bs,
                                   msq, msk, ot, ct, cr, om, cm, wl, wr, lq,
                                   lk, ho, hc) {
    throw std::runtime_error("Sparse metadata unavailable");
  }

  void operator()(Tensor, const Tensor, const Tensor, const Tensor,
                  const Tensor, const Tensor, int64_t, int64_t, int64_t,
                  int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                  int64_t, int64_t, int64_t, int64_t, std::string,
                  std::string, bool, bool) const override {
    throw std::runtime_error("Sparse metadata unavailable");
  }
};

#endif

}  // namespace infini::ops

#endif
