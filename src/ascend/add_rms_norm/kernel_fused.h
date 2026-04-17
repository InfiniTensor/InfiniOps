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
// Computes x_out = x1 + x2 and y_out = rms_norm(x_out, gamma, eps) in a
// single CANN launch.  The fused API has higher host-side launch overhead
// (~200 us) compared to the decomposed `aclnnAdd` + `aclnnRmsNorm` path (~39
// us), but may offer better NPU-side efficiency for large tensors where kernel
// fusion reduces memory traffic.
//
// Select via `implementation_index=1` in Python:
//   infini.ops.add_rms_norm(..., implementation_index=1, stream=s)
template <>
class Operator<AddRmsNorm, Device::Type::kAscend, 1> : public AddRmsNorm {
 public:
  Operator(const Tensor x1, const Tensor x2, const Tensor gamma, float eps,
           Tensor y_out, Tensor x_out)
      : AddRmsNorm(x1, x2, gamma, eps, y_out, x_out),
        x1_cache_(x1),
        x2_cache_(x2),
        gamma_cache_(gamma),
        y_out_cache_(y_out),
        x_out_cache_(x_out) {
    // `aclnnAddRmsNorm` requires `rstdOut` to have the same ndim as x1, with
    // the last gamma.ndim() dimensions set to 1.  For example:
    //   x1 shape(2, 32, 128), gamma shape(128) -> rstdOut shape(2, 32, 1)
    //   x1 shape(64, 128),    gamma shape(128) -> rstdOut shape(64, 1)
    fused_rstd_shape_.reserve(ndim_);
    for (size_t i = 0; i < ndim_ - gamma.ndim(); ++i) {
      fused_rstd_shape_.push_back(static_cast<int64_t>(x1.size(i)));
    }
    for (size_t i = 0; i < gamma.ndim(); ++i) {
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
    if (!ascend::isAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    x1_cache_.release();
    x2_cache_.release();
    gamma_cache_.release();
    y_out_cache_.release();
    x_out_cache_.release();

    // `rstd_tensor_` leaks with the executor at shutdown (see `64c367c`).
    if (rstd_data_) aclrtFree(rstd_data_);
  }

  void operator()(const Tensor x1, const Tensor x2, const Tensor gamma,
                  float eps, Tensor y_out, Tensor x_out) const override {
    auto t_x1 = x1_cache_.get(const_cast<void*>(x1.data()));
    auto t_x2 = x2_cache_.get(const_cast<void*>(x2.data()));
    auto t_gamma = gamma_cache_.get(const_cast<void*>(gamma.data()));
    auto t_y_out = y_out_cache_.get(y_out.data());
    auto t_x_out = x_out_cache_.get(x_out.data());
    auto stream = static_cast<aclrtStream>(stream_);

    if (!executor_) {
      aclnnAddRmsNormGetWorkspaceSize(
          t_x1, t_x2, t_gamma, static_cast<double>(eps), t_y_out, rstd_tensor_,
          t_x_out, &ws_size_, &executor_);
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_x1, const_cast<void*>(x1.data()));
      aclSetInputTensorAddr(executor_, 1, t_x2, const_cast<void*>(x2.data()));
      aclSetInputTensorAddr(executor_, 2, t_gamma,
                            const_cast<void*>(gamma.data()));
      aclSetOutputTensorAddr(executor_, 0, t_y_out, y_out.data());
      // rstd at output index 1 has a stable address — no update needed.
      aclSetOutputTensorAddr(executor_, 2, t_x_out, x_out.data());
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);
    aclnnAddRmsNorm(arena.buf, ws_size_, executor_, stream);
  }

 private:
  mutable ascend::AclTensorCache x1_cache_;

  mutable ascend::AclTensorCache x2_cache_;

  mutable ascend::AclTensorCache gamma_cache_;

  mutable ascend::AclTensorCache y_out_cache_;

  mutable ascend::AclTensorCache x_out_cache_;

  std::vector<int64_t> fused_rstd_shape_;

  void* rstd_data_ = nullptr;

  aclTensor* rstd_tensor_ = nullptr;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;
};

}  // namespace infini::ops

#endif
