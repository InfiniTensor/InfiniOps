#ifndef INFINI_OPS_ASCEND_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ASCEND_RMS_NORM_KERNEL_H_

#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_rms_norm.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/rms_norm.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kAscend> : public RmsNorm {
 public:
  Operator(const Tensor input, const Tensor weight, float eps, Tensor out)
      : RmsNorm(input, weight, eps, out),
        in_cache_(input),
        weight_cache_(weight),
        out_cache_(out) {
    // aclnnRmsNorm writes rstd as a required side output.
    // Size computed here; buffer obtained from pool in `operator()`.
    rstd_shape_ = {static_cast<int64_t>(batch_size_),
                   static_cast<int64_t>(nhead_)};
    rstd_size_ = batch_size_ * nhead_ * sizeof(float);
  }

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    in_cache_.release();
    weight_cache_.release();
    out_cache_.release();
    // `rstd_tensor_` leaks with the executor at shutdown (see `64c367c`).
  }

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    auto t_in = in_cache_.get(const_cast<void*>(input.data()));
    auto t_weight = weight_cache_.get(const_cast<void*>(weight.data()));
    auto t_out = out_cache_.get(out.data());
    auto stream = static_cast<aclrtStream>(stream_);

    // Obtain shared rstd buffer from pool.
    auto& rstd_arena =
        ascend::GetWorkspacePool().Ensure(stream, rstd_size_, "temp");

    // Lazily create rstd tensor descriptor on first call.
    if (!rstd_tensor_) {
      rstd_tensor_ = aclCreateTensor(rstd_shape_.data(), 2, ACL_FLOAT,
                                     /*strides=*/nullptr, 0, ACL_FORMAT_ND,
                                     rstd_shape_.data(), 2, rstd_arena.buf);
    } else {
      aclSetRawTensorAddr(rstd_tensor_, rstd_arena.buf);
    }

    if (!executor_) {
      aclnnRmsNormGetWorkspaceSize(t_in, t_weight, eps, t_out, rstd_tensor_,
                                   &ws_size_, &executor_);
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_in,
                            const_cast<void*>(input.data()));
      aclSetInputTensorAddr(executor_, 1, t_weight,
                            const_cast<void*>(weight.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
      aclSetOutputTensorAddr(executor_, 1, rstd_tensor_, rstd_arena.buf);
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);
    aclnnRmsNorm(arena.buf, ws_size_, executor_, stream);
  }

 private:
  mutable ascend::AclTensorCache in_cache_;

  mutable ascend::AclTensorCache weight_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;

  std::vector<int64_t> rstd_shape_;

  uint64_t rstd_size_ = 0;

  mutable aclTensor* rstd_tensor_ = nullptr;
};

}  // namespace infini::ops

#include "ascend/rms_norm/kernel_custom.h"

#endif
