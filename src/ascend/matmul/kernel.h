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
      : Matmul(a, b, c, trans_a, trans_b) {}

  void operator()(const Tensor a, const Tensor b, Tensor c, bool trans_a,
                  bool trans_b) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_a = ascend::buildAclTensor(a, trans_a);
    auto t_b = ascend::buildAclTensor(b, trans_b);
    auto t_out = ascend::buildAclTensor(c);

    uint64_t ws_needed = 0;
    aclOpExecutor* executor = nullptr;
    // cube_math_type = 1: allow fp16 accumulation.
    int8_t cube_math_type = 1;
    aclnnMatmulGetWorkspaceSize(t_a, t_b, t_out, cube_math_type, &ws_needed,
                                &executor);
    auto& arena = ascend::workspacePool().ensure(stream, ws_needed);
    aclnnMatmul(arena.buf, ws_needed, executor, stream);

    aclDestroyTensor(t_a);
    aclDestroyTensor(t_b);
    aclDestroyTensor(t_out);
  }
};

}  // namespace infini::ops

#endif
