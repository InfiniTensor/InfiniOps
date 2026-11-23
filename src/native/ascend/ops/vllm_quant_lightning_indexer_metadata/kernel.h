#ifndef INFINI_OPS_ASCEND_VLLM_QUANT_LIGHTNING_INDEXER_METADATA_KERNEL_H_
#define INFINI_OPS_ASCEND_VLLM_QUANT_LIGHTNING_INDEXER_METADATA_KERNEL_H_

#include <stdexcept>
#include <string>

#include "base/vllm_quant_lightning_indexer_metadata.h"
#include "operator.h"

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_vllm_quant_lightning_indexer_metadata.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#endif

namespace infini::ops {

#ifdef INFINI_HAS_ASCEND_CUSTOM_TRANSFORMER

template <>
class Operator<VllmQuantLightningIndexerMetadata, Device::Type::kAscend, 0>
    : public VllmQuantLightningIndexerMetadata {
 public:
  static bool supports(Tensor output, const Tensor q, const Tensor k, int64_t,
                       int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                       int64_t, std::string layout_q, std::string layout_k,
                       int64_t, int64_t, int64_t, int64_t, int64_t) {
    return output.device().type() == Device::Type::kAscend &&
           q.device() == output.device() && k.device() == output.device() &&
           output.dtype() == DataType::kInt32 &&
           output.shape() == Tensor::Shape{1024} && q.ndim() == 1 &&
           k.ndim() == 1 && q.dtype() == DataType::kInt32 &&
           k.dtype() == DataType::kInt32 && q.IsContiguous() &&
           k.IsContiguous() && output.IsContiguous() && layout_q == "TND" &&
           layout_k == "PA_BSND";
  }

  Operator(Tensor output, const Tensor q, const Tensor k, int64_t nhq,
           int64_t nhk, int64_t hd, int64_t qq, int64_t kq, int64_t bs,
           int64_t msq, int64_t msk, std::string lq, std::string lk,
           int64_t sc, int64_t sm, int64_t pt, int64_t nt, int64_t cr)
      : VllmQuantLightningIndexerMetadata(output, q, k, nhq, nhk, hd, qq, kq,
                                          bs, msq, msk, lq, lk, sc, sm, pt,
                                          nt, cr),
        nhq_(nhq),
        nhk_(nhk),
        hd_(hd),
        qq_(qq),
        kq_(kq),
        bs_(bs),
        msq_(msq),
        msk_(msk),
        lq_(std::move(lq)),
        lk_(std::move(lk)),
        sc_(sc),
        sm_(sm),
        pt_(pt),
        nt_(nt),
        cr_(cr),
        q_cache_(q),
        k_cache_(k),
        output_cache_(output) {
    if (!supports(output, q, k, nhq, nhk, hd, qq, kq, bs, msq, msk, lq_,
                  lk_, sc, sm, pt, nt, cr)) {
      throw std::invalid_argument("unsupported Ascend QLI metadata operands");
    }
  }

  ~Operator() override {
    if (executor_ != nullptr) {
      q_cache_.release();
      k_cache_.release();
      output_cache_.release();
    }
  }

  bool capture_safe() const override { return false; }

  void operator()(Tensor output, const Tensor q, const Tensor k, int64_t nhq,
                  int64_t nhk, int64_t hd, int64_t qq, int64_t kq, int64_t bs,
                  int64_t msq, int64_t msk, std::string lq, std::string lk,
                  int64_t sc, int64_t sm, int64_t pt, int64_t nt,
                  int64_t cr) const override {
    if (!supports(output, q, k, nhq, nhk, hd, qq, kq, bs, msq, msk, lq, lk,
                  sc, sm, pt, nt, cr)) {
      throw std::invalid_argument("unsupported Ascend QLI metadata operands");
    }

    auto* t_q = q_cache_.get(const_cast<void*>(q.data()));
    auto* t_k = k_cache_.get(const_cast<void*>(k.data()));
    auto* t_output = output_cache_.get(output.data());
    if (executor_ == nullptr) {
      ascend::CheckAcl(
          aclnnVllmQuantLightningIndexerMetadataGetWorkspaceSize(
              t_q, t_k, nhq_, nhk_, hd_, qq_, kq_, bs_, msq_, msk_,
              const_cast<char*>(lq_.c_str()), const_cast<char*>(lk_.c_str()),
              sc_, sm_, pt_, nt_, cr_, t_output, &workspace_size_, &executor_),
          "QLI metadata workspace query");
      ascend::CheckAcl(aclSetAclOpExecutorRepeatable(executor_),
                       "QLI metadata repeatable executor setup");
    } else {
      ascend::CheckAcl(AclSetInputTensorAddr(executor_, 0, t_q, const_cast<void*>(q.data())),
                       "QLI metadata query address update");
      ascend::CheckAcl(AclSetInputTensorAddr(executor_, 1, t_k, const_cast<void*>(k.data())),
                       "QLI metadata key address update");
      ascend::CheckAcl(
          AclSetOutputTensorAddr(executor_, 0, t_output, output.data()),
          "QLI metadata output address update");
    }

    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, workspace_size_, "qli_metadata");
    ascend::CheckAcl(aclnnVllmQuantLightningIndexerMetadata(
                         arena.buf, workspace_size_, executor_, stream),
                     "QLI metadata execution");
  }

 private:
   int64_t nhq_;
  int64_t nhk_;
  int64_t hd_;
  int64_t qq_;
  int64_t kq_;
  int64_t bs_;
  int64_t msq_;
  int64_t msk_;
  std::string lq_;
  std::string lk_;
  int64_t sc_;
  int64_t sm_;
  int64_t pt_;
  int64_t nt_;
  int64_t cr_;
  mutable ascend::AclTensorCache q_cache_;
  mutable ascend::AclTensorCache k_cache_;
  mutable ascend::AclTensorCache output_cache_;
  mutable uint64_t workspace_size_ = 0;
  mutable aclOpExecutor* executor_ = nullptr;
};

#else

template <>
class Operator<VllmQuantLightningIndexerMetadata, Device::Type::kAscend, 0>
    : public VllmQuantLightningIndexerMetadata {
 public:
  template <typename... Args>
  static bool supports(const Args&...) {
    return false;
  }

  Operator(Tensor output, const Tensor q, const Tensor k, int64_t nhq,
           int64_t nhk, int64_t hd, int64_t qq, int64_t kq, int64_t bs,
           int64_t msq, int64_t msk, std::string lq, std::string lk,
           int64_t sc, int64_t sm, int64_t pt, int64_t nt, int64_t cr)
      : VllmQuantLightningIndexerMetadata(output, q, k, nhq, nhk, hd, qq, kq,
                                          bs, msq, msk, lq, lk, sc, sm, pt,
                                          nt, cr) {
    throw std::runtime_error("QLI metadata requires custom transformer");
  }

  void operator()(Tensor, const Tensor, const Tensor, int64_t, int64_t,
                  int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                  std::string, std::string, int64_t, int64_t, int64_t,
                  int64_t, int64_t) const override {
    throw std::runtime_error("QLI metadata unavailable");
  }
};

#endif

}  // namespace infini::ops

#endif
