#ifndef INFINI_OPS_ASCEND_MATMUL_KERNEL_H_
#define INFINI_OPS_ASCEND_MATMUL_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_matmul.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/matmul.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Matmul, Device::Type::kAscend> : public Matmul {
 public:
  Operator(const Tensor a, const Tensor b, Tensor c, bool trans_a, bool trans_b)
      : Matmul(a, b, c, trans_a, trans_b),
        a_cache_(a, trans_a),
        b_cache_(b, trans_b),
        out_cache_(c) {}

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    a_cache_.release();
    b_cache_.release();
    out_cache_.release();
  }

  void operator()(const Tensor a, const Tensor b, Tensor c, bool trans_a,
                  bool trans_b) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_a = a_cache_.get(const_cast<void*>(a.data()));
    auto t_b = b_cache_.get(const_cast<void*>(b.data()));
    auto t_out = out_cache_.get(c.data());

    if (!executor_) {
      int8_t cube_math_type = 1;
      aclnnMatmulGetWorkspaceSize(t_a, t_b, t_out, cube_math_type, &ws_size_,
                                  &executor_);
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_a, const_cast<void*>(a.data()));
      aclSetInputTensorAddr(executor_, 1, t_b, const_cast<void*>(b.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, c.data());
    }

    auto& arena = ascend::workspacePool().ensure(stream, ws_size_);
    aclnnMatmul(arena.buf, ws_size_, executor_, stream);
  }

 private:
  mutable ascend::AclTensorCache a_cache_;

  mutable ascend::AclTensorCache b_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;
};

}  // namespace infini::ops

#endif
