#ifndef INFINI_OPS_ASCEND_TOPK_TOPP_SAMPLING_KERNEL_ATB_H_
#define INFINI_OPS_ASCEND_TOPK_TOPP_SAMPLING_KERNEL_ATB_H_

#ifdef INFINI_HAS_ATB

#include <cstddef>
#include <cstdint>

#include "acl/acl.h"
#include "aclnn_fill_scalar.h"
#include "aclnn_sim_thread_exponential.h"
#include "atb/context.h"
#include "atb/infer_op_params.h"
#include "atb/operation.h"
#include "atb/types.h"
#include "base/topk_topp_sampling.h"
#include "native/ascend/atb_common_.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

// ATB-based fused top-k/top-p sampling via `atb::infer::TopkToppSamplingParam`
// (implementation index 0).
//
// Uses `BATCH_TOPK_EXPONENTIAL_SAMPLING` which matches vLLM's Gumbel-trick
// sampling semantics (`q.exponential_()` -> `probs.div(q).argmax()`).
// Exponential sampling does not require `randSeeds`, making the ATB operation
// parameter-stable and cacheable across calls with the same `topk`.
//
// ATB constraint: input probabilities must be float16 or bfloat16.
// The caller must cast float32 probs to float16 before invoking this kernel.
//
// ATB tensor layout (from `atb_ops_info.ini`):
//   in0 (probs)      : [B, V] float16/bf16
//   in1 (seeds)      : [B, 1] int32
//   in2 (top_p)      : [B, 1] float16/bf16
//   in3 (exp_random) : [B, V] float16/bf16
//   out0 (indices)   : [B, 1] int32
//   out1 (out_probs) : [B, 1] float16/bf16
template <>
class Operator<TopkToppSampling, Device::Type::kAscend, 0>
    : public TopkToppSampling {
 public:
  Operator(const Tensor probs, int64_t topk, double topp, Tensor out)
      : TopkToppSampling(probs, topk, topp, out) {
    atb::infer::TopkToppSamplingParam param;
    param.topkToppSamplingType =
        atb::infer::TopkToppSamplingParam::BATCH_TOPK_EXPONENTIAL_SAMPLING;
    param.topk = static_cast<uint32_t>(topk_);

    atb::Status s = atb::CreateOperation(param, &op_);

    if (s != atb::NO_ERROR) {
      fprintf(stderr,
              "[TopkToppSampling] atb::CreateOperation failed (status=%d)\n",
              static_cast<int>(s));
    }

    seeds_cache_ = ascend::AclTensorCache(
        {static_cast<int64_t>(batch_size_), 1}, ACL_INT32, nullptr);
    top_p_cache_ =
        ascend::AclTensorCache({static_cast<int64_t>(batch_size_), 1},
                               ascend::ToAclDtype(dtype_), nullptr);
    exp_random_cache_ = ascend::AclTensorCache(
        {static_cast<int64_t>(batch_size_), static_cast<int64_t>(vocab_size_)},
        ascend::ToAclDtype(dtype_), nullptr);
    zero_i32_ = aclCreateScalar(&zero_i32_storage_, ACL_INT32);
    top_p_scalar_ = aclCreateScalar(&top_p_storage_, ACL_FLOAT);
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    seeds_cache_.release();
    top_p_cache_.release();
    exp_random_cache_.release();

    if (zero_i32_) aclDestroyScalar(zero_i32_);
    if (top_p_scalar_) aclDestroyScalar(top_p_scalar_);
    if (op_) atb::DestroyOperation(op_);
  }

  Operator(const Operator&) = delete;

  Operator& operator=(const Operator&) = delete;

  void operator()(const Tensor probs, int64_t topk, double topp,
                  Tensor out) const override {
    if (!op_) return;

    auto stream = static_cast<aclrtStream>(stream_);
    atb::Context* ctx = ascend::GetAtbContext(stream);

    int64_t B = batch_size_;
    int64_t V = vocab_size_;
    aclDataType probs_dt = ascend::ToAclDtype(probs.dtype());
    uint64_t probs_elem = 2;  // Float16 or bf16 — both 2 bytes.
    void* probs_ptr = const_cast<void*>(probs.data());
    void* out_ptr = out.data();

    // Auxiliary buffers: seeds [B,1] int32 + in2 [B,1] fp16 + out1 [B,1] fp16.
    // Also allocate in3 [B,V] fp16 as a scratch buffer.
    uint64_t seeds_bytes = static_cast<uint64_t>(B) * 4;
    uint64_t in2_bytes = static_cast<uint64_t>(B) * probs_elem;
    uint64_t out1_bytes = static_cast<uint64_t>(B) * probs_elem;
    uint64_t in3_bytes = static_cast<uint64_t>(B * V) * probs_elem;
    uint64_t aux_bytes = seeds_bytes + in2_bytes + out1_bytes + in3_bytes;

    // Build tensors using raw descriptors.
    auto mk2d = [](aclDataType dt, int64_t d0, int64_t d1, void* data,
                   uint64_t elem_sz) -> atb::Tensor {
      atb::Tensor t;
      t.desc.dtype = dt;
      t.desc.format = ACL_FORMAT_ND;
      t.desc.shape.dimNum = 2;
      t.desc.shape.dims[0] = d0;
      t.desc.shape.dims[1] = d1;
      t.deviceData = data;
      t.dataSize = static_cast<uint64_t>(d0 * d1) * elem_sz;

      return t;
    };

    // Ensure workspace covers both auxiliary buffers and ATB's own workspace.
    auto& arena = ascend::GetWorkspacePool().Ensure(stream, aux_bytes);
    auto* base = static_cast<uint8_t*>(arena.buf);
    void* seeds_ptr = base;
    void* top_p_ptr = base + seeds_bytes;
    void* in3_ptr = base + seeds_bytes + in2_bytes;
    void* out1_ptr = base + seeds_bytes + in2_bytes + in3_bytes;

    FillAuxTensors(stream, seeds_ptr, top_p_ptr, in3_ptr);

    atb::Tensor t_probs = mk2d(probs_dt, B, V, probs_ptr, probs_elem);
    atb::Tensor t_seeds = mk2d(ACL_INT32, B, 1, seeds_ptr, 4);
    atb::Tensor t_in2 = mk2d(probs_dt, B, 1, top_p_ptr, probs_elem);
    atb::Tensor t_in3 = mk2d(probs_dt, B, V, in3_ptr, probs_elem);
    atb::Tensor t_out0 = mk2d(ACL_INT32, B, 1, out_ptr, 4);
    atb::Tensor t_out1 = mk2d(probs_dt, B, 1, out1_ptr, probs_elem);

    atb::VariantPack vp;
    vp.inTensors = {t_probs, t_seeds, t_in2, t_in3};
    vp.outTensors = {t_out0, t_out1};

    uint64_t ws_size = 0;
    atb::Status s = op_->Setup(vp, ws_size, ctx);

    if (s != atb::NO_ERROR) {
      fprintf(stderr, "[TopkToppSampling] Setup failed (status=%d)\n",
              static_cast<int>(s));

      return;
    }

    // ATB workspace (separate from auxiliary buffers).
    uint8_t* ws_ptr = nullptr;

    if (ws_size > 0) {
      auto& ws_arena =
          ascend::GetWorkspacePool().Ensure(stream, aux_bytes + ws_size);

      // Re-derive auxiliary pointers from the (possibly reallocated) arena.
      base = static_cast<uint8_t*>(ws_arena.buf);
      ws_ptr = base + aux_bytes;

      // Update tensor data pointers in case the arena was reallocated.
      seeds_ptr = base;
      top_p_ptr = base + seeds_bytes;
      in3_ptr = base + seeds_bytes + in2_bytes;
      out1_ptr = base + seeds_bytes + in2_bytes + in3_bytes;

      FillAuxTensors(stream, seeds_ptr, top_p_ptr, in3_ptr);

      vp.inTensors[1].deviceData = seeds_ptr;
      vp.inTensors[2].deviceData = top_p_ptr;
      vp.inTensors[3].deviceData = in3_ptr;
      vp.outTensors[1].deviceData = out1_ptr;

      // Re-run Setup with updated pointers.
      s = op_->Setup(vp, ws_size, ctx);

      if (s != atb::NO_ERROR) {
        fprintf(stderr, "[TopkToppSampling] Setup (retry) failed (status=%d)\n",
                static_cast<int>(s));

        return;
      }
    }

    s = op_->Execute(vp, ws_ptr, ws_size, ctx);

    if (s != atb::NO_ERROR) {
      fprintf(stderr, "[TopkToppSampling] Execute failed (status=%d)\n",
              static_cast<int>(s));
    }
  }

 private:
  void FillAuxTensors(aclrtStream stream, void* seeds_ptr, void* top_p_ptr,
                      void* exp_random_ptr) const {
    auto t_seeds = seeds_cache_.get(seeds_ptr);
    auto t_top_p = top_p_cache_.get(top_p_ptr);
    auto t_exp_random = exp_random_cache_.get(exp_random_ptr);

    FillScalar(stream, t_seeds, seeds_ptr, zero_i32_, seeds_fill_ws_,
               seeds_fill_exec_);
    FillScalar(stream, t_top_p, top_p_ptr, top_p_scalar_, top_p_fill_ws_,
               top_p_fill_exec_);
    FillExponential(stream, t_exp_random, exp_random_ptr);
  }

  void FillScalar(aclrtStream stream, aclTensor* tensor, void* data,
                  aclScalar* value, uint64_t& workspace_size,
                  aclOpExecutor*& executor) const {
    if (!executor) {
      aclnnInplaceFillScalarGetWorkspaceSize(tensor, value, &workspace_size,
                                             &executor);
      aclSetAclOpExecutorRepeatable(executor);
    } else {
      aclSetInputTensorAddr(executor, 0, tensor, data);
    }

    auto& arena =
        ascend::GetWorkspacePool().Ensure(stream, workspace_size, "topk_fill");
    aclnnInplaceFillScalar(arena.buf, workspace_size, executor, stream);
  }

  void FillExponential(aclrtStream stream, aclTensor* tensor,
                       void* data) const {
    if (!exp_exec_) {
      aclnnSimThreadExponentialGetWorkspaceSize(
          tensor, static_cast<int64_t>(batch_size_ * vocab_size_), 1.0,
          /*seed=*/0, /*offset=*/0, &exp_ws_, &exp_exec_);
      aclSetAclOpExecutorRepeatable(exp_exec_);
    } else {
      aclSetInputTensorAddr(exp_exec_, 0, tensor, data);
    }

    auto& arena =
        ascend::GetWorkspacePool().Ensure(stream, exp_ws_, "topk_exp");
    aclnnSimThreadExponential(arena.buf, exp_ws_, exp_exec_, stream);
  }

  mutable ascend::AclTensorCache seeds_cache_;

  mutable ascend::AclTensorCache top_p_cache_;

  mutable ascend::AclTensorCache exp_random_cache_;

  int32_t zero_i32_storage_{0};

  float top_p_storage_{static_cast<float>(topp_)};

  aclScalar* zero_i32_ = nullptr;

  aclScalar* top_p_scalar_ = nullptr;

  mutable aclOpExecutor* seeds_fill_exec_ = nullptr;

  mutable uint64_t seeds_fill_ws_ = 0;

  mutable aclOpExecutor* top_p_fill_exec_ = nullptr;

  mutable uint64_t top_p_fill_ws_ = 0;

  mutable aclOpExecutor* exp_exec_ = nullptr;

  mutable uint64_t exp_ws_ = 0;

  atb::Operation* op_ = nullptr;
};

}  // namespace infini::ops

#endif  // INFINI_HAS_ATB

#endif  // INFINI_OPS_ASCEND_TOPK_TOPP_SAMPLING_KERNEL_ATB_H_
