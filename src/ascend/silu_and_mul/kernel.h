#ifndef INFINI_OPS_ASCEND_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_ASCEND_SILU_AND_MUL_KERNEL_H_

#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_copy.h"
#include "aclnnop/aclnn_swi_glu.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/silu_and_mul.h"
#include "operator.h"

namespace infini::ops {

// Calls `aclnnSwiGlu` directly on the concatenated `x = [gate, up]` tensor.
//
// `aclnnSwiGlu` splits `x` along `dim` into `[first_half, second_half]` and
// computes `second_half * silu(first_half)`, i.e. `up * silu(gate)`.
//
// `aclnnSwiGlu` ignores output strides and writes contiguously.  When the
// output is non-contiguous, a contiguous staging buffer is used and the
// result is copied back via `aclnnInplaceCopy`.
template <>
class Operator<SiluAndMul, Device::Type::kAscend, 0> : public SiluAndMul {
 public:
  Operator(const Tensor x, int64_t dim, Tensor out)
      : SiluAndMul(x, dim, out), x_cache_(x), out_cache_(out) {
    needs_copy_ = !is_out_contiguous_;

    if (needs_copy_) {
      out_staging_size_ = out.numel() * kDataTypeToSize.at(out.dtype());
    }
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.  Inputs and
    // outputs are referenced by the Repeatable executors (`swiglu_exec_`,
    // `copy_exec_`); releasing them here prevents `~AclTensorCache()` from
    // double-freeing at shutdown.
    x_cache_.release();
    out_cache_.release();

    // The staging cache is held by `swiglu_exec_` / `copy_exec_`; release to
    // avoid double-free on destruction.
    if (out_staging_cache_) out_staging_cache_->release();
  }

  void operator()(const Tensor x, int64_t dim, Tensor out) const override {
    auto t_x = x_cache_.get(const_cast<void*>(x.data()));
    auto t_out = out_cache_.get(out.data());
    auto stream = static_cast<aclrtStream>(stream_);

    // Determine effective output target.
    aclTensor* t_swiglu_out = t_out;
    void* swiglu_out_data = out.data();

    if (needs_copy_) {
      auto& staging = ascend::GetWorkspacePool().Ensure(
          stream, out_staging_size_, "staging");

      if (!out_staging_cache_) {
        std::vector<int64_t> out_shape(out_shape_.begin(), out_shape_.end());
        out_staging_cache_.emplace(out_shape, ascend::ToAclDtype(out_dtype_),
                                   staging.buf);
      }

      t_swiglu_out = out_staging_cache_->get(staging.buf);
      swiglu_out_data = staging.buf;
    }

    // Call `aclnnSwiGlu`.
    if (!swiglu_exec_) {
      aclnnSwiGluGetWorkspaceSize(t_x, dim_, t_swiglu_out, &swiglu_ws_,
                                  &swiglu_exec_);
      aclSetAclOpExecutorRepeatable(swiglu_exec_);
    } else {
      aclSetInputTensorAddr(swiglu_exec_, 0, t_x, const_cast<void*>(x.data()));
      aclSetOutputTensorAddr(swiglu_exec_, 0, t_swiglu_out, swiglu_out_data);
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, swiglu_ws_);
    aclnnSwiGlu(arena.buf, swiglu_ws_, swiglu_exec_, stream);

    // Copy staging buffer back to non-contiguous output if needed.
    if (needs_copy_) {
      if (!copy_exec_) {
        aclnnInplaceCopyGetWorkspaceSize(t_out, t_swiglu_out, &copy_ws_,
                                         &copy_exec_);
        aclSetAclOpExecutorRepeatable(copy_exec_);
      } else {
        aclSetInputTensorAddr(copy_exec_, 0, t_out, out.data());
        aclSetInputTensorAddr(copy_exec_, 1, t_swiglu_out, swiglu_out_data);
      }

      auto& copy_arena = ascend::GetWorkspacePool().Ensure(stream, copy_ws_);
      aclnnInplaceCopy(copy_arena.buf, copy_ws_, copy_exec_, stream);
    }
  }

 private:
  mutable ascend::AclTensorCache x_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable std::optional<ascend::AclTensorCache> out_staging_cache_;

  bool needs_copy_ = false;

  uint64_t out_staging_size_ = 0;

  mutable aclOpExecutor* swiglu_exec_ = nullptr;

  mutable uint64_t swiglu_ws_ = 0;

  mutable aclOpExecutor* copy_exec_ = nullptr;

  mutable uint64_t copy_ws_ = 0;
};

}  // namespace infini::ops

#endif
