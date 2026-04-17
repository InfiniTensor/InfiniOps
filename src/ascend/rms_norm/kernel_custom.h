#ifndef INFINI_OPS_ASCEND_RMS_NORM_KERNEL_CUSTOM_H_
#define INFINI_OPS_ASCEND_RMS_NORM_KERNEL_CUSTOM_H_

#ifdef INFINI_HAS_CUSTOM_KERNELS

#include <algorithm>
#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_cast.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/rms_norm.h"
#include "operator.h"

// Forward-declare the generated AscendC kernel launch function.
// This symbol is provided by the `no_workspace_kernel` static library
// built from `ascend/custom_kernel/csrc/ops/rms_norm/op_kernel/rms_norm.cpp`
// via `ascendc_library()`.
extern "C" uint32_t aclrtlaunch_rms_norm(
    uint32_t blockDim, void* stream, void* x, void* weight, void* y,
    int64_t totalRows, int64_t dimLength, int64_t dimLengthAlign,
    int64_t formerNum, int64_t formerLength, int64_t tailLength, float eps,
    int64_t dtypeSize);

namespace infini::ops {

// Custom AscendC fused RmsNorm kernel (implementation index 1).
//
// A single-kernel implementation that computes RMSNorm in one launch, avoiding
// the 5-sub-op decomposition of `aclnnRmsNorm` (index 0).  Uses `Sqrt` +
// scalar division instead of `Rsqrt` for higher precision (~1e-7 fp32 error
// vs ~0.2% with `Rsqrt`).
//
// Select via `implementation_index=1` in Python:
//   infini.ops.rms_norm(input, weight, eps, out, implementation_index=1,
//                       stream=s)
//
// Requirements:
//   - Input last dimension must be 32-byte aligned (divisible by 16 for fp16
//     or 8 for fp32).  All standard LLM hidden dimensions satisfy this.
//   - Weight must have the same dtype as input.
//   - The custom kernel binary must be linked (`BUILD_CUSTOM_KERNEL=ON`).
template <>
class Operator<RmsNorm, Device::Type::kAscend, 1> : public RmsNorm {
 public:
  Operator(const Tensor input, const Tensor weight, float eps, Tensor out)
      : RmsNorm(input, weight, eps, out) {
    // Dtype size in bytes.
    dtype_size_ = (input.dtype() == DataType::kFloat16) ? 2 : 4;

    // Alignment check (32-byte boundary).
    int64_t align_elems = 32 / dtype_size_;
    dim_length_align_ =
        ((static_cast<int64_t>(dim_) + align_elems - 1) / align_elems) *
        align_elems;
    assert(static_cast<int64_t>(dim_) == dim_length_align_ &&
           "Custom RmsNorm kernel requires 32-byte aligned last dimension");

    total_rows_ =
        static_cast<int64_t>(batch_size_) * static_cast<int64_t>(nhead_);

    // For fp16 input, weight needs fp32 conversion because the custom
    // kernel always reads weight as fp32.
    needs_weight_cast_ = (dtype_size_ == 2);

    if (needs_weight_cast_) {
      // Allocate persistent fp32 weight buffer on device.
      size_t fp32_bytes = static_cast<size_t>(dim_) * sizeof(float);
      aclrtMalloc(&weight_fp32_data_, fp32_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

      // AclTensorCache for the cast source (fp16 weight descriptor).
      weight_src_cache_ = ascend::AclTensorCache({static_cast<int64_t>(dim_)},
                                                 ACL_FLOAT16, nullptr);

      // AclTensorCache for the cast destination (fp32 weight buffer).
      weight_dst_cache_ = ascend::AclTensorCache({static_cast<int64_t>(dim_)},
                                                 ACL_FLOAT, weight_fp32_data_);
    }
  }

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    weight_src_cache_.release();
    weight_dst_cache_.release();

    if (weight_fp32_data_) aclrtFree(weight_fp32_data_);
  }

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    // Determine fp32 weight pointer.
    void* weight_fp32;

    if (needs_weight_cast_) {
      // Cast weight fp16 -> fp32 using cached ACLNN executor.
      auto t_src = weight_src_cache_.get(const_cast<void*>(weight.data()));
      auto t_dst = weight_dst_cache_.get(weight_fp32_data_);

      if (!cast_exec_) {
        aclnnCastGetWorkspaceSize(t_src, ACL_FLOAT, t_dst, &cast_ws_,
                                  &cast_exec_);
        aclSetAclOpExecutorRepeatable(cast_exec_);
      } else {
        aclSetInputTensorAddr(cast_exec_, 0, t_src,
                              const_cast<void*>(weight.data()));
        aclSetOutputTensorAddr(cast_exec_, 0, t_dst, weight_fp32_data_);
      }

      auto& arena = ascend::workspacePool().ensure(stream, cast_ws_);
      aclnnCast(arena.buf, cast_ws_, cast_exec_, stream);
      weight_fp32 = weight_fp32_data_;
    } else {
      // Input is fp32 — weight is already fp32.
      weight_fp32 = const_cast<void*>(weight.data());
    }

    // Block-level tiling: distribute rows across cores.
    // Maximum block dimension covers Ascend 910B (20-40 AIV cores).
    // Over-subscribing is safe (runtime multiplexes blocks across cores),
    // though slightly sub-optimal due to per-block weight loading.
    static constexpr int64_t kMaxBlockDim = 40;
    int64_t used_cores = std::min(total_rows_, kMaxBlockDim);
    int64_t former_length = (total_rows_ + used_cores - 1) / used_cores;
    int64_t tail_length = former_length - 1;
    int64_t former_num = total_rows_ - tail_length * used_cores;
    uint32_t block_dim = static_cast<uint32_t>(used_cores);

    // Launch custom AscendC kernel.
    aclrtlaunch_rms_norm(
        block_dim, stream, const_cast<void*>(input.data()), weight_fp32,
        out.data(), total_rows_, static_cast<int64_t>(dim_), dim_length_align_,
        former_num, former_length, tail_length, eps, dtype_size_);
  }

 private:
  int64_t dtype_size_;

  int64_t dim_length_align_;

  int64_t total_rows_;

  bool needs_weight_cast_;

  void* weight_fp32_data_ = nullptr;

  mutable ascend::AclTensorCache weight_src_cache_;

  mutable ascend::AclTensorCache weight_dst_cache_;

  mutable aclOpExecutor* cast_exec_ = nullptr;

  mutable uint64_t cast_ws_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_HAS_CUSTOM_KERNELS
#endif  // INFINI_OPS_ASCEND_RMS_NORM_KERNEL_CUSTOM_H_
