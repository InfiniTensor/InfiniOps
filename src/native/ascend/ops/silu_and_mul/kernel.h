#ifndef INFINI_OPS_ASCEND_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_ASCEND_SILU_AND_MUL_KERNEL_H_

#include <algorithm>
#include <optional>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_copy.h"
#include "aclnnop/aclnn_swi_glu.h"
#include "base/silu_and_mul.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

#ifdef INFINI_HAS_CUSTOM_KERNELS
#include "aclrtlaunch_silu_and_mul_tiled.h"
#endif

namespace infini::ops {

// `aclnnSwiGlu` splits the last input dimension into `[gate, up]` and
// computes `silu(gate) * up`. CANN writes its output contiguously, so a
// staging tensor is required when the InfiniOps output has padding.
template <>
class Operator<SiluAndMul, Device::Type::kAscend> : public SiluAndMul {
 public:
  Operator(const Tensor input, Tensor out)
      : SiluAndMul(input, out), input_cache_(input), out_cache_(out) {
#ifdef INFINI_HAS_CUSTOM_KERNELS
    // Padded outputs avoid the ACLNN staging and copy; contiguous outputs
    // retain the vendor kernel's lower launch overhead.
    use_tiled_kernel_ =
        !is_out_contiguous_ && (ndim_ == 2 || ndim_ == 3) && hidden_size_ > 0 &&
        hidden_size_ <= 8192 && hidden_size_ % 128 == 0 &&
        input.stride(-1) == 1 && out.stride(-1) == 1 &&
        input.stride(-2) >= static_cast<int64_t>(hidden_size_ * 2) &&
        out.stride(-2) >= static_cast<int64_t>(hidden_size_) &&
        (ndim_ == 2 || (input.stride(0) == static_cast<int64_t>(input.size(1)) *
                                               input.stride(1) &&
                        out.stride(0) == static_cast<int64_t>(out.size(1)) *
                                             out.stride(1))) &&
        input_type_ == DataType::kBFloat16;
#endif

    needs_copy_ = !is_out_contiguous_;
    if (needs_copy_) {
      out_staging_size_ = out.numel() * kDataTypeToSize.at(out.dtype());
    }
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    input_cache_.release();
    out_cache_.release();
    if (out_staging_cache_) out_staging_cache_->release();
  }

  void operator()(const Tensor input, Tensor out) const override {
#ifdef INFINI_HAS_CUSTOM_KERNELS
    if (use_tiled_kernel_) {
      auto rows = output_size_ / hidden_size_;
      if (rows == 0) return;

      auto blocks = static_cast<uint32_t>(std::min<Tensor::Size>(rows, 8) * 8);
      auto ret = aclrtlaunch_silu_and_mul_tiled(
          blocks, static_cast<aclrtStream>(stream_),
          const_cast<void*>(input.data()), out.data(),
          static_cast<int64_t>(rows), static_cast<int64_t>(hidden_size_),
          static_cast<int64_t>(input.stride(-2)),
          static_cast<int64_t>(out.stride(-2)));
      assert(ret == ACL_SUCCESS &&
             "Ascend tiled `SiluAndMul` execution failed");
      return;
    }
#endif

    auto t_input = input_cache_.get(const_cast<void*>(input.data()));
    auto t_out = out_cache_.get(out.data());
    auto stream = static_cast<aclrtStream>(stream_);

    aclTensor* t_swiglu_out = t_out;
    void* swiglu_out_data = out.data();

    if (needs_copy_) {
      auto& staging = ascend::GetWorkspacePool().Ensure(
          stream, out_staging_size_, "silu_and_mul_output");

      if (!out_staging_cache_) {
        std::vector<int64_t> out_shape(out_shape_.begin(), out_shape_.end());
        out_staging_cache_.emplace(out_shape, ascend::ToAclDtype(out_type_),
                                   staging.buf);
      }

      t_swiglu_out = out_staging_cache_->get(staging.buf);
      swiglu_out_data = staging.buf;
    }

    if (!swiglu_exec_) {
      aclnnSwiGluGetWorkspaceSize(t_input, -1, t_swiglu_out, &swiglu_ws_,
                                  &swiglu_exec_);
      aclSetAclOpExecutorRepeatable(swiglu_exec_);
    } else {
      aclSetInputTensorAddr(swiglu_exec_, 0, t_input,
                            const_cast<void*>(input.data()));
      aclSetOutputTensorAddr(swiglu_exec_, 0, t_swiglu_out, swiglu_out_data);
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, swiglu_ws_);
    aclnnSwiGlu(arena.buf, swiglu_ws_, swiglu_exec_, stream);

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
  mutable ascend::AclTensorCache input_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable std::optional<ascend::AclTensorCache> out_staging_cache_;

  bool use_tiled_kernel_{false};

  bool needs_copy_{false};

  uint64_t out_staging_size_{0};

  mutable aclOpExecutor* swiglu_exec_{nullptr};

  mutable uint64_t swiglu_ws_{0};

  mutable aclOpExecutor* copy_exec_{nullptr};

  mutable uint64_t copy_ws_{0};
};

}  // namespace infini::ops

#endif
