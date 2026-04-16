#ifndef INFINI_OPS_ASCEND_ADD_KERNEL_H_
#define INFINI_OPS_ASCEND_ADD_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_add.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/add.h"
#include "data_type.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kAscend> : public Add {
 public:
  Operator(const Tensor input, const Tensor other, Tensor out)
      : Add(input, other, out),
        in_cache_(input),
        oth_cache_(other),
        out_cache_(out) {
    // `aclCreateScalar` stores the pointer rather than copying the value, so
    // `alpha_storage_*` must remain alive for the lifetime of `alpha_`.
    // The alpha scalar type must match the tensor dtype: use int64 for integer
    // dtypes and float for floating-point dtypes.
    if (ascend::isIntegerDtype(input.dtype())) {
      alpha_ = aclCreateScalar(&alpha_int_storage_, ACL_INT64);
    } else {
      alpha_ = aclCreateScalar(&alpha_float_storage_, ACL_FLOAT);
    }
  }

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;

    // Test: destroy tensors first, then executor.
    // If CANN executor reference-counts tensors, this is safe.
    // If not, aclDestroyAclOpExecutor will double-free and crash.
    in_cache_.destroy();
    oth_cache_.destroy();
    out_cache_.destroy();

    if (executor_) aclDestroyAclOpExecutor(executor_);
    if (alpha_) aclDestroyScalar(alpha_);
  }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_in = in_cache_.get(const_cast<void*>(input.data()));
    auto t_oth = oth_cache_.get(const_cast<void*>(other.data()));
    auto t_out = out_cache_.get(out.data());

    if (!executor_) {
      aclnnAddGetWorkspaceSize(t_in, t_oth, alpha_, t_out, &ws_size_,
                               &executor_);
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_in,
                            const_cast<void*>(input.data()));
      aclSetInputTensorAddr(executor_, 1, t_oth,
                            const_cast<void*>(other.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
    }

    auto& arena = ascend::workspacePool().ensure(stream, ws_size_);
    aclnnAdd(arena.buf, ws_size_, executor_, stream);
  }

 private:
  mutable ascend::AclTensorCache in_cache_;

  mutable ascend::AclTensorCache oth_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;

  float alpha_float_storage_ =
      1.0f;  // Stable address for `aclCreateScalar` (float).
  int64_t alpha_int_storage_ =
      1;  // Stable address for `aclCreateScalar` (int).
  aclScalar* alpha_ = nullptr;
};

}  // namespace infini::ops

#endif
