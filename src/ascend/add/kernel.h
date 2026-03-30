#ifndef INFINI_OPS_ASCEND_ADD_KERNEL_H_
#define INFINI_OPS_ASCEND_ADD_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_add.h"
#include "ascend/device.h"
#include "base/add.h"
#include "data_type.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kAscend> : public Add {
 public:
  Operator(const Tensor input, const Tensor other, Tensor out)
      : Add(input, other, out) {
    // aclCreateScalar stores the pointer rather than copying the value, so
    // alpha_storage_* must remain alive for the lifetime of alpha_.
    // The alpha scalar type must match the tensor dtype: use int64 for integer
    // dtypes and float for floating-point dtypes.
    if (ascend::isIntegerDtype(input.dtype())) {
        alpha_ = aclCreateScalar(&alpha_int_storage_, ACL_INT64);
    } else {
        alpha_ = aclCreateScalar(&alpha_float_storage_, ACL_FLOAT);
    }
  }

  ~Operator() {
    aclDestroyScalar(alpha_);
  }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_in  = ascend::buildAclTensor(input);
    auto t_oth = ascend::buildAclTensor(other);
    auto t_out = ascend::buildAclTensor(out);
    uint64_t ws_needed = 0;
    aclOpExecutor* executor = nullptr;
    aclnnAddGetWorkspaceSize(t_in, t_oth, alpha_, t_out, &ws_needed, &executor);
    auto& arena = ascend::workspacePool().ensure(stream, ws_needed);
    aclnnAdd(arena.buf, ws_needed, executor, stream);
    aclDestroyTensor(t_in);
    aclDestroyTensor(t_oth);
    aclDestroyTensor(t_out);
  }

 private:
  float      alpha_float_storage_ = 1.0f;  // stable address for aclCreateScalar (float)
  int64_t    alpha_int_storage_   = 1;     // stable address for aclCreateScalar (int)
  aclScalar* alpha_ = nullptr;
};

}  // namespace infini::ops

#endif
