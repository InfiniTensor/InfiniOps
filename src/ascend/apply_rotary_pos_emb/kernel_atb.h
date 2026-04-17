#ifndef INFINI_OPS_ASCEND_APPLY_ROTARY_POS_EMB_KERNEL_ATB_H_
#define INFINI_OPS_ASCEND_APPLY_ROTARY_POS_EMB_KERNEL_ATB_H_

#ifdef INFINI_HAS_ATB

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "ascend/atb_common_.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "atb/context.h"
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "atb/types.h"
#include "base/apply_rotary_pos_emb.h"
#include "operator.h"

namespace infini::ops {

// Apply-only rotary embedding via ATB `RopeParam` (implementation index 1).
//
// Takes pre-gathered `[T, D]` cos/sin tensors directly — no `IndexSelect`.
// ATB Rope with `rotaryCoeff=2`, `cosFormat=0` expects:
//   inTensors:  Q `[T, hiddenQ]`, K `[T, hiddenK]`, cos `[T, D]`,
//               sin `[T, D]`, seqlen `[1]`.
//   outTensors: Q_out `[T, hiddenQ]`, K_out `[T, hiddenK]`.
//
// Restrictions:
//   - `is_neox_style` must be true (rotaryCoeff=2).
//   - fp16 only (ATB inference constraint).
template <>
class Operator<ApplyRotaryPosEmb, Device::Type::kAscend, 1>
    : public ApplyRotaryPosEmb {
 public:
  Operator(const Tensor query, const Tensor key, const Tensor cos,
           const Tensor sin, int64_t head_size, bool is_neox_style,
           Tensor query_out, Tensor key_out)
      : ApplyRotaryPosEmb(query, key, cos, sin, head_size, is_neox_style,
                          query_out, key_out) {
    assert(is_neox_style &&
           "ATB `ApplyRotaryPosEmb` requires neox style (rotaryCoeff=2)");

    const int64_t T = num_tokens_;
    const int64_t D = head_size_;
    int64_t hiddenQ = static_cast<int64_t>(query.numel()) / T;
    int64_t hiddenK = static_cast<int64_t>(key.numel()) / T;

    q_2d_shape_ = {T, hiddenQ};
    k_2d_shape_ = {T, hiddenK};
    cos_sin_shape_ = {T, D};
    seqlen_shape_ = {1};
    acl_dt_ = ascend::toAclDtype(query.dtype());
    elem_size_ = static_cast<uint64_t>(query.element_size());

    // Allocate seqlen buffer: 1 int32 element holding T.
    aclrtMalloc(&seqlen_dev_, sizeof(int32_t), ACL_MEM_MALLOC_NORMAL_ONLY);
    int32_t seqlen_val = static_cast<int32_t>(T);
    aclrtMemcpy(seqlen_dev_, sizeof(int32_t), &seqlen_val, sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE);

    // Create ATB Rope operation.
    atb::infer::RopeParam param;
    param.rotaryCoeff = 2;
    param.cosFormat = 0;
    atb::Status s = atb::CreateOperation(param, &op_);

    assert(s == atb::NO_ERROR && "atb::CreateOperation(Rope) failed");
  }

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;

    if (op_) atb::DestroyOperation(op_);
    if (seqlen_dev_) aclrtFree(seqlen_dev_);
  }

  Operator(const Operator&) = delete;

  Operator& operator=(const Operator&) = delete;

  void operator()(const Tensor query, const Tensor key, const Tensor cos,
                  const Tensor sin, int64_t head_size, bool is_neox_style,
                  Tensor query_out, Tensor key_out) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    int64_t T = query.size(0);
    int64_t D = head_size;
    int64_t hiddenQ = static_cast<int64_t>(query.numel()) / T;
    int64_t hiddenK = static_cast<int64_t>(key.numel()) / T;

    // Copy q→q_out, k→k_out if not inplace.
    size_t elem_sz = query.element_size();

    if (query.data() != query_out.data()) {
      aclrtMemcpyAsync(query_out.data(),
                       static_cast<size_t>(T * hiddenQ) * elem_sz, query.data(),
                       static_cast<size_t>(T * hiddenQ) * elem_sz,
                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }

    if (key.data() != key_out.data()) {
      aclrtMemcpyAsync(key_out.data(),
                       static_cast<size_t>(T * hiddenK) * elem_sz, key.data(),
                       static_cast<size_t>(T * hiddenK) * elem_sz,
                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }

    // Build ATB VariantPack: 5 inputs + 2 outputs.
    atb::Context* ctx = ascend::getAtbContext(stream);

    uint64_t q_bytes = static_cast<uint64_t>(T * hiddenQ) * elem_size_;
    uint64_t k_bytes = static_cast<uint64_t>(T * hiddenK) * elem_size_;
    uint64_t cs_bytes = static_cast<uint64_t>(T * D) * elem_size_;

    atb::Tensor t_q =
        ascend::toAtbTensor(q_2d_shape_, acl_dt_, query_out.data(), q_bytes);
    atb::Tensor t_k =
        ascend::toAtbTensor(k_2d_shape_, acl_dt_, key_out.data(), k_bytes);
    atb::Tensor t_cos = ascend::toAtbTensor(
        cos_sin_shape_, acl_dt_, const_cast<void*>(cos.data()), cs_bytes);
    atb::Tensor t_sin = ascend::toAtbTensor(
        cos_sin_shape_, acl_dt_, const_cast<void*>(sin.data()), cs_bytes);
    atb::Tensor t_seqlen =
        ascend::toAtbTensor(seqlen_shape_, ACL_INT32, seqlen_dev_,
                            static_cast<uint64_t>(sizeof(int32_t)));

    atb::VariantPack vp;
    vp.inTensors = {t_q, t_k, t_cos, t_sin, t_seqlen};
    vp.outTensors = {t_q, t_k};

    uint64_t ws_size = 0;
    atb::Status s = op_->Setup(vp, ws_size, ctx);

    assert(s == atb::NO_ERROR && "ATB Rope Setup failed");

    uint8_t* ws_ptr = nullptr;

    if (ws_size > 0) {
      auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size);
      ws_ptr = static_cast<uint8_t*>(arena.buf);
    }

    s = op_->Execute(vp, ws_ptr, ws_size, ctx);

    assert(s == atb::NO_ERROR && "ATB Rope Execute failed");
  }

 private:
  atb::Operation* op_ = nullptr;

  void* seqlen_dev_ = nullptr;

  std::vector<int64_t> q_2d_shape_;

  std::vector<int64_t> k_2d_shape_;

  std::vector<int64_t> cos_sin_shape_;

  std::vector<int64_t> seqlen_shape_;

  aclDataType acl_dt_ = ACL_DT_UNDEFINED;

  uint64_t elem_size_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_HAS_ATB

#endif  // INFINI_OPS_ASCEND_APPLY_ROTARY_POS_EMB_KERNEL_ATB_H_
