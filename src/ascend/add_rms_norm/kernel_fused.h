#ifndef INFINI_OPS_ASCEND_ADD_RMS_NORM_KERNEL_FUSED_H_
#define INFINI_OPS_ASCEND_ADD_RMS_NORM_KERNEL_FUSED_H_

#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_add_rms_norm.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/add_rms_norm.h"
#include "operator.h"

namespace infini::ops {

// Fused implementation via `aclnnAddRmsNorm` (implementation index 1).
//
// Computes `residual_out = input + other` and `out = rms_norm(residual_out,
// weight, eps)` in a single CANN launch.  The fused API has higher host-side
// launch overhead (~200 us) compared to the decomposed `aclnnAdd` +
// `aclnnRmsNorm` path (~39 us), but may offer better NPU-side efficiency for
// large tensors where kernel fusion reduces memory traffic.
//
// Select via `implementation_index=1` in Python:
//   infini.ops.add_rms_norm(..., implementation_index=1, stream=s)
template <>
class Operator<AddRmsNorm, Device::Type::kAscend, 1> : public AddRmsNorm {
 public:
  Operator(const Tensor input, const Tensor other, const Tensor weight,
           float eps, Tensor out, Tensor residual_out)
      : AddRmsNorm(input, other, weight, eps, out, residual_out),
        input_cache_(input),
        other_cache_(other),
        weight_cache_(weight),
        out_cache_(out),
        residual_out_cache_(residual_out) {
    // `aclnnAddRmsNorm` requires `rstdOut` to have the same ndim as `input`,
    // with the last `weight.ndim()` dimensions set to 1.  For example:
    //   `input` (2, 32, 128), `weight` (128) -> `rstdOut` (2, 32, 1).
    //   `input` (64, 128),    `weight` (128) -> `rstdOut` (64, 1).
    fused_rstd_shape_.reserve(ndim_);
    for (size_t i = 0; i < ndim_ - weight.ndim(); ++i) {
      fused_rstd_shape_.push_back(static_cast<int64_t>(input.size(i)));
    }
    for (size_t i = 0; i < weight.ndim(); ++i) {
      fused_rstd_shape_.push_back(1);
    }

    size_t rstd_elems = 1;
    for (auto d : fused_rstd_shape_) {
      rstd_elems *= static_cast<size_t>(d);
    }
    size_t rstd_bytes = rstd_elems * sizeof(float);
    aclrtMalloc(&rstd_data_, rstd_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

    rstd_tensor_ = aclCreateTensor(
        fused_rstd_shape_.data(),
        static_cast<int64_t>(fused_rstd_shape_.size()), ACL_FLOAT,
        /*strides=*/nullptr, 0, ACL_FORMAT_ND, fused_rstd_shape_.data(),
        static_cast<int64_t>(fused_rstd_shape_.size()), rstd_data_);
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    input_cache_.release();
    other_cache_.release();
    weight_cache_.release();
    out_cache_.release();
    residual_out_cache_.release();

    // `rstd_tensor_` leaks with the executor at shutdown (see `64c367c`).
    if (rstd_data_) aclrtFree(rstd_data_);
  }

  void operator()(const Tensor input, const Tensor other, const Tensor weight,
                  float eps, Tensor out, Tensor residual_out) const override {
    auto t_input = input_cache_.get(const_cast<void*>(input.data()));
    auto t_other = other_cache_.get(const_cast<void*>(other.data()));
    auto t_weight = weight_cache_.get(const_cast<void*>(weight.data()));
    auto t_out = out_cache_.get(out.data());
    auto t_residual_out = residual_out_cache_.get(residual_out.data());
    auto stream = static_cast<aclrtStream>(stream_);

    if (!executor_) {
      aclnnAddRmsNormGetWorkspaceSize(
          t_input, t_other, t_weight, static_cast<double>(eps), t_out,
          rstd_tensor_, t_residual_out, &ws_size_, &executor_);
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_input,
                            const_cast<void*>(input.data()));
      aclSetInputTensorAddr(executor_, 1, t_other,
                            const_cast<void*>(other.data()));
      aclSetInputTensorAddr(executor_, 2, t_weight,
                            const_cast<void*>(weight.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
      // `rstd` at output index 1 has a stable address — no update needed.
      aclSetOutputTensorAddr(executor_, 2, t_residual_out, residual_out.data());
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);
    aclnnAddRmsNorm(arena.buf, ws_size_, executor_, stream);
  }

 private:
  mutable ascend::AclTensorCache input_cache_;

  mutable ascend::AclTensorCache other_cache_;

  mutable ascend::AclTensorCache weight_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable ascend::AclTensorCache residual_out_cache_;

  std::vector<int64_t> fused_rstd_shape_;

  void* rstd_data_ = nullptr;

  aclTensor* rstd_tensor_ = nullptr;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;
};

}  // namespace infini::ops

#endif
