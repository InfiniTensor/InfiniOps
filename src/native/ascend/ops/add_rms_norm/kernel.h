#ifndef INFINI_OPS_ASCEND_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ASCEND_ADD_RMS_NORM_KERNEL_H_

#include <cassert>
#include <optional>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_add.h"
#include "aclnn_add_rms_norm.h"
#include "aclnn_rms_norm.h"
#include "base/add_rms_norm.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

// Decomposed implementation: `aclnnAdd` + `aclnnRmsNorm`.
//
// The fused `aclnnAddRmsNorm` API has ~200 us host-side launch overhead that
// dominates small-tensor dispatch.  Decomposing into two fast ACLNN calls
// reduces host dispatch from ~224 us to ~56 us (4x faster) with negligible
// NPU-side impact for inference tensor sizes.
template <>
class Operator<AddRmsNorm, Device::Type::kAscend, 0> : public AddRmsNorm {
 public:
  Operator(const Tensor input, const Tensor residual, const Tensor weight,
           std::optional<float> eps, Tensor out, Tensor residual_out)
      : AddRmsNorm(input, residual, weight, eps, out, residual_out),
        input_cache_(input),
        residual_cache_(residual),
        weight_cache_(weight),
        out_cache_(out),
        residual_out_cache_(residual_out) {
    assert(out.IsContiguous() &&
           "`AddRmsNorm` Ascend path requires contiguous `out`.");

    // Alpha scalar for `aclnnAdd` (`residual_out = input + 1.0 * residual`).
    alpha_ = aclCreateScalar(&alpha_storage_, ACL_FLOAT);

    // `aclnnRmsNorm` writes `rstd` as a required side output.  Size is
    // computed here; the buffer is obtained from the pool in `operator()`.
    rstd_shape_ = {static_cast<int64_t>(batch_size_),
                   static_cast<int64_t>(nhead_)};
    rstd_size_ = batch_size_ * nhead_ * sizeof(float);
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    input_cache_.release();
    residual_cache_.release();
    weight_cache_.release();
    out_cache_.release();
    residual_out_cache_.release();

    // `rstd_tensor_` leaks with `norm_exec_` at shutdown (see `64c367c`).
    if (alpha_) aclDestroyScalar(alpha_);
  }

  void operator()(const Tensor input, const Tensor residual,
                  const Tensor weight, std::optional<float> eps, Tensor out,
                  Tensor residual_out) const override {
    auto resolved_eps = eps.value_or(eps_);
    auto t_input = input_cache_.get(const_cast<void*>(input.data()));
    auto t_residual = residual_cache_.get(const_cast<void*>(residual.data()));
    auto t_weight = weight_cache_.get(const_cast<void*>(weight.data()));
    auto t_out = out_cache_.get(out.data());
    auto t_residual_out = residual_out_cache_.get(residual_out.data());
    auto stream = static_cast<aclrtStream>(stream_);

    // Step 1: `residual_out = input + residual`.
    if (!add_exec_) {
      aclnnAddGetWorkspaceSize(t_input, t_residual, alpha_, t_residual_out,
                               &add_ws_, &add_exec_);
      aclSetAclOpExecutorRepeatable(add_exec_);
    } else {
      aclSetInputTensorAddr(add_exec_, 0, t_input,
                            const_cast<void*>(input.data()));
      aclSetInputTensorAddr(add_exec_, 1, t_residual,
                            const_cast<void*>(residual.data()));
      aclSetOutputTensorAddr(add_exec_, 0, t_residual_out, residual_out.data());
    }
    auto& add_arena = ascend::GetWorkspacePool().Ensure(stream, add_ws_);
    aclnnAdd(add_arena.buf, add_ws_, add_exec_, stream);

    // Obtain shared `rstd` buffer from pool.
    auto& rstd_arena =
        ascend::GetWorkspacePool().Ensure(stream, rstd_size_, "temp");

    // Lazily create the `rstd` tensor descriptor on first call.
    if (!rstd_tensor_) {
      rstd_tensor_ = aclCreateTensor(rstd_shape_.data(), 2, ACL_FLOAT,
                                     /*strides=*/nullptr, 0, ACL_FORMAT_ND,
                                     rstd_shape_.data(), 2, rstd_arena.buf);
    } else {
      aclSetRawTensorAddr(rstd_tensor_, rstd_arena.buf);
    }

    // Step 2: `out = rms_norm(residual_out, weight, eps)`.
    if (!norm_exec_) {
      aclnnRmsNormGetWorkspaceSize(t_residual_out, t_weight, resolved_eps,
                                   t_out, rstd_tensor_, &norm_ws_, &norm_exec_);
      aclSetAclOpExecutorRepeatable(norm_exec_);
    } else {
      aclSetInputTensorAddr(norm_exec_, 0, t_residual_out, residual_out.data());
      aclSetInputTensorAddr(norm_exec_, 1, t_weight,
                            const_cast<void*>(weight.data()));
      aclSetOutputTensorAddr(norm_exec_, 0, t_out, out.data());
      aclSetOutputTensorAddr(norm_exec_, 1, rstd_tensor_, rstd_arena.buf);
    }
    auto& norm_arena = ascend::GetWorkspacePool().Ensure(stream, norm_ws_);
    aclnnRmsNorm(norm_arena.buf, norm_ws_, norm_exec_, stream);
  }

 private:
  mutable ascend::AclTensorCache input_cache_;

  mutable ascend::AclTensorCache residual_cache_;

  mutable ascend::AclTensorCache weight_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable ascend::AclTensorCache residual_out_cache_;

  float alpha_storage_ = 1.0f;

  aclScalar* alpha_ = nullptr;

  std::vector<int64_t> rstd_shape_;

  uint64_t rstd_size_ = 0;

  mutable aclTensor* rstd_tensor_ = nullptr;

  mutable aclOpExecutor* add_exec_ = nullptr;

  mutable uint64_t add_ws_ = 0;

  mutable aclOpExecutor* norm_exec_ = nullptr;

  mutable uint64_t norm_ws_ = 0;
};

template <>
class Operator<AddRmsNorm, Device::Type::kAscend, 1> : public AddRmsNorm {
 public:
  Operator(const Tensor input, const Tensor residual, const Tensor weight,
           std::optional<float> eps, Tensor out, Tensor residual_out)
      : AddRmsNorm(input, residual, weight, eps, out, residual_out),
        input_cache_(input),
        residual_cache_(residual),
        weight_cache_(weight),
        out_cache_(out),
        residual_out_cache_(residual_out) {
    assert(input.IsContiguous() && residual.IsContiguous() &&
           out.IsContiguous() && residual_out.IsContiguous() &&
           "`aclnnAddRmsNorm` Ascend path requires contiguous tensors.");

    // `aclnnAddRmsNorm` requires `rstdOut` to have the same ndim as `input`,
    // with the normalized dimensions set to 1.
    rstd_shape_.reserve(ndim_);
    for (Tensor::Size i = 0; i < ndim_ - weight.ndim(); ++i) {
      rstd_shape_.push_back(static_cast<int64_t>(input.size(i)));
    }
    for (Tensor::Size i = 0; i < weight.ndim(); ++i) {
      rstd_shape_.push_back(1);
    }

    Tensor::Size rstd_elems = 1;
    for (auto dim : rstd_shape_) {
      rstd_elems *= static_cast<Tensor::Size>(dim);
    }
    auto rstd_bytes = rstd_elems * sizeof(float);
    auto ret = aclrtMalloc(&rstd_data_, rstd_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    assert(ret == ACL_SUCCESS &&
           "`aclnnAddRmsNorm` Ascend path failed to allocate `rstdOut`.");

    rstd_tensor_ = aclCreateTensor(
        rstd_shape_.data(), static_cast<int64_t>(rstd_shape_.size()), ACL_FLOAT,
        /*strides=*/nullptr, 0, ACL_FORMAT_ND, rstd_shape_.data(),
        static_cast<int64_t>(rstd_shape_.size()), rstd_data_);
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    input_cache_.release();
    residual_cache_.release();
    weight_cache_.release();
    out_cache_.release();
    residual_out_cache_.release();

    // `rstd_tensor_` leaks with the executor at shutdown (see `64c367c`).
    if (rstd_data_) aclrtFree(rstd_data_);
  }

  void operator()(const Tensor input, const Tensor residual,
                  const Tensor weight, std::optional<float> eps, Tensor out,
                  Tensor residual_out) const override {
    auto resolved_eps = static_cast<double>(eps.value_or(eps_));
    auto t_input = input_cache_.get(const_cast<void*>(input.data()));
    auto t_residual = residual_cache_.get(const_cast<void*>(residual.data()));
    auto t_weight = weight_cache_.get(const_cast<void*>(weight.data()));
    auto t_out = out_cache_.get(out.data());
    auto t_residual_out = residual_out_cache_.get(residual_out.data());
    auto stream = static_cast<aclrtStream>(stream_);

    if (!executor_) {
      aclnnAddRmsNormGetWorkspaceSize(t_input, t_residual, t_weight,
                                      resolved_eps, t_out, rstd_tensor_,
                                      t_residual_out, &ws_, &executor_);
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_input,
                            const_cast<void*>(input.data()));
      aclSetInputTensorAddr(executor_, 1, t_residual,
                            const_cast<void*>(residual.data()));
      aclSetInputTensorAddr(executor_, 2, t_weight,
                            const_cast<void*>(weight.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
      // `rstd` at output index 1 has a stable address.
      aclSetOutputTensorAddr(executor_, 2, t_residual_out, residual_out.data());
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_);
    aclnnAddRmsNorm(arena.buf, ws_, executor_, stream);
  }

 private:
  mutable ascend::AclTensorCache input_cache_;

  mutable ascend::AclTensorCache residual_cache_;

  mutable ascend::AclTensorCache weight_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable ascend::AclTensorCache residual_out_cache_;

  std::vector<int64_t> rstd_shape_;

  void* rstd_data_ = nullptr;

  aclTensor* rstd_tensor_ = nullptr;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_ = 0;
};

}  // namespace infini::ops

#endif
