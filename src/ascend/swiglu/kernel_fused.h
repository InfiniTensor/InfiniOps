#ifndef INFINI_OPS_ASCEND_SWIGLU_KERNEL_FUSED_H_
#define INFINI_OPS_ASCEND_SWIGLU_KERNEL_FUSED_H_

#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_copy.h"
#include "aclnnop/aclnn_cat.h"
#include "aclnnop/aclnn_swi_glu.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/swiglu.h"
#include "operator.h"

namespace infini::ops {

// Fused implementation via `aclnnSwiGlu` (implementation index 1).
//
// Concatenates `[gate, input]` into a temp buffer via `aclnnCat`, then calls
// `aclnnSwiGlu` which computes `second_half * silu(first_half)` in a single
// fused kernel, i.e. `input * silu(gate)`.
//
// This trades an extra `aclnnCat` launch for a single fused SwiGLU kernel
// instead of separate `aclnnSilu` + `aclnnMul`.  The net benefit is one fewer
// intermediate buffer materialised on-device (the silu temp is eliminated).
//
// `aclnnSwiGlu` requires a contiguous output tensor.  When the caller's output
// is non-contiguous, a contiguous temp buffer is used and the result is copied
// back via `aclnnInplaceCopy`.
//
// Select via `implementation_index=1` in Python:
//   infini.ops.swiglu(..., implementation_index=1, stream=s)
template <>
class Operator<Swiglu, Device::Type::kAscend, 1> : public Swiglu {
 public:
  Operator(const Tensor input, const Tensor gate, Tensor out)
      : Swiglu(input, gate, out),
        gate_cache_(gate),
        in_cache_(input),
        out_cache_(out) {
    // Compute the concatenated shape: same as input but with last dim doubled.
    cat_shape_.assign(input.shape().begin(), input.shape().end());
    cat_shape_.back() *= 2;

    uint64_t cat_elems = 1;

    for (auto d : cat_shape_) {
      cat_elems *= static_cast<uint64_t>(d);
    }

    cat_size_ = cat_elems * kDataTypeToSize.at(input.dtype());

    // `aclnnSwiGlu` ignores output strides and writes contiguously.
    // When the output is non-contiguous we need a contiguous staging buffer.
    needs_copy_ = !is_out_contiguous_;

    if (needs_copy_) {
      out_staging_size_ = output_size_ * kDataTypeToSize.at(out.dtype());
    }
  }

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    gate_cache_.release();
    in_cache_.release();
    out_cache_.release();

    if (cat_tensor_list_) aclDestroyTensorList(cat_tensor_list_);
  }

  void operator()(const Tensor input, const Tensor gate,
                  Tensor out) const override {
    auto t_gate = gate_cache_.get(const_cast<void*>(gate.data()));
    auto t_in = in_cache_.get(const_cast<void*>(input.data()));
    auto t_out = out_cache_.get(out.data());
    auto stream = static_cast<aclrtStream>(stream_);

    // Obtain shared temp buffer for the concatenated tensor.
    auto& cat_arena = ascend::workspacePool().ensure(stream, cat_size_, "temp");

    // Lazily build the cat output tensor cache on first call.
    if (!cat_out_cache_) {
      cat_out_cache_.emplace(cat_shape_, ascend::toAclDtype(input_type_),
                             cat_arena.buf);
    }

    auto t_cat = cat_out_cache_->get(cat_arena.buf);

    // Step 1: cat([gate, input], dim=-1) -> cat_buf.
    if (!cat_exec_) {
      aclTensor* tensors[2] = {t_gate, t_in};
      cat_tensor_list_ =
          aclCreateTensorList(const_cast<const aclTensor**>(tensors), 2);
      aclnnCatGetWorkspaceSize(cat_tensor_list_,
                               static_cast<int64_t>(ndim_ - 1), t_cat, &cat_ws_,
                               &cat_exec_);
      aclSetAclOpExecutorRepeatable(cat_exec_);
    } else {
      // The tensor list references the same `aclTensor*` objects whose data
      // pointers were already updated by `get()` above.
      aclSetOutputTensorAddr(cat_exec_, 0, t_cat, cat_arena.buf);
    }

    auto& cat_ws_arena = ascend::workspacePool().ensure(stream, cat_ws_);
    aclnnCat(cat_ws_arena.buf, cat_ws_, cat_exec_, stream);

    // Step 2: swiglu(cat_buf, dim=-1) -> out (or staging buffer).
    aclTensor* t_swiglu_out = t_out;
    void* swiglu_out_data = out.data();

    if (needs_copy_) {
      auto& staging =
          ascend::workspacePool().ensure(stream, out_staging_size_, "staging");

      if (!out_staging_cache_) {
        std::vector<int64_t> out_shape(out_shape_.begin(), out_shape_.end());
        out_staging_cache_.emplace(out_shape, ascend::toAclDtype(out_type_),
                                   staging.buf);
      }

      t_swiglu_out = out_staging_cache_->get(staging.buf);
      swiglu_out_data = staging.buf;
    }

    if (!swiglu_exec_) {
      aclnnSwiGluGetWorkspaceSize(t_cat, static_cast<int64_t>(ndim_ - 1),
                                  t_swiglu_out, &swiglu_ws_, &swiglu_exec_);
      aclSetAclOpExecutorRepeatable(swiglu_exec_);
    } else {
      aclSetInputTensorAddr(swiglu_exec_, 0, t_cat, cat_arena.buf);
      aclSetOutputTensorAddr(swiglu_exec_, 0, t_swiglu_out, swiglu_out_data);
    }

    auto& swiglu_arena = ascend::workspacePool().ensure(stream, swiglu_ws_);
    aclnnSwiGlu(swiglu_arena.buf, swiglu_ws_, swiglu_exec_, stream);

    // Step 3 (non-contiguous output only): copy staging -> out.
    if (needs_copy_) {
      if (!copy_exec_) {
        aclnnInplaceCopyGetWorkspaceSize(t_out, t_swiglu_out, &copy_ws_,
                                         &copy_exec_);
        aclSetAclOpExecutorRepeatable(copy_exec_);
      } else {
        aclSetInputTensorAddr(copy_exec_, 0, t_out, out.data());
        aclSetInputTensorAddr(copy_exec_, 1, t_swiglu_out, swiglu_out_data);
      }

      auto& copy_arena = ascend::workspacePool().ensure(stream, copy_ws_);
      aclnnInplaceCopy(copy_arena.buf, copy_ws_, copy_exec_, stream);
    }
  }

 private:
  mutable ascend::AclTensorCache gate_cache_;

  mutable ascend::AclTensorCache in_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable std::optional<ascend::AclTensorCache> cat_out_cache_;

  mutable std::optional<ascend::AclTensorCache> out_staging_cache_;

  std::vector<int64_t> cat_shape_;

  uint64_t cat_size_ = 0;

  bool needs_copy_ = false;

  uint64_t out_staging_size_ = 0;

  mutable aclTensorList* cat_tensor_list_ = nullptr;

  mutable aclOpExecutor* cat_exec_ = nullptr;

  mutable uint64_t cat_ws_ = 0;

  mutable aclOpExecutor* swiglu_exec_ = nullptr;

  mutable uint64_t swiglu_ws_ = 0;

  mutable aclOpExecutor* copy_exec_ = nullptr;

  mutable uint64_t copy_ws_ = 0;
};

}  // namespace infini::ops

#endif
