#ifndef INFINI_OPS_ASCEND_SWIGLU_KERNEL_H_
#define INFINI_OPS_ASCEND_SWIGLU_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_mul.h"
#include "aclnn_silu.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/swiglu.h"
#include "data_type.h"
#include "operator.h"

namespace infini::ops {

// Implements SwiGLU as two ACLNN calls: silu(gate) into a temp buffer,
// then elementwise mul(input, temp) into out.
// aclnnSiluMul was not used because it fuses silu_AND_mul on the same
// tensor (x * silu(x)), whereas SwiGLU requires input * silu(gate) —
// two distinct inputs.
template <>
class Operator<Swiglu, Device::Type::kAscend> : public Swiglu {
 public:
  Operator(const Tensor input, const Tensor gate, Tensor out)
      : Swiglu(input, gate, out) {
    size_t nbytes = input.numel() * kDataTypeToSize.at(input.dtype());
    aclrtMalloc(&temp_buf_, nbytes, ACL_MEM_MALLOC_NORMAL_ONLY);
  }

  ~Operator() { aclrtFree(temp_buf_); }

  void operator()(const Tensor input, const Tensor gate,
                  Tensor out) const override {
    // temp_buf_ is a contiguous scratch buffer; give it contiguous strides.
    Tensor temp_t{temp_buf_, gate.shape(), gate.dtype(), gate.device()};

    auto t_in = ascend::buildAclTensor(input);
    auto t_gate = ascend::buildAclTensor(gate);
    auto t_out = ascend::buildAclTensor(out);
    auto t_temp = ascend::buildAclTensor(temp_t);

    uint64_t ws_needed = 0;
    aclOpExecutor* exec = nullptr;
    auto stream = static_cast<aclrtStream>(stream_);

    // Step 1: silu(gate) -> temp.  SwiGLU = input * silu(gate).
    aclnnSiluGetWorkspaceSize(t_gate, t_temp, &ws_needed, &exec);
    auto& silu_arena = ascend::workspacePool().ensure(stream, ws_needed);
    aclnnSilu(silu_arena.buf, ws_needed, exec, stream);

    // Step 2: mul(input, temp) -> out.
    uint64_t mul_ws = 0;
    exec = nullptr;
    aclnnMulGetWorkspaceSize(t_in, t_temp, t_out, &mul_ws, &exec);
    auto& mul_arena = ascend::workspacePool().ensure(stream, mul_ws);
    aclnnMul(mul_arena.buf, mul_ws, exec, stream);

    aclDestroyTensor(t_in);
    aclDestroyTensor(t_gate);
    aclDestroyTensor(t_out);
    aclDestroyTensor(t_temp);
  }

 private:
  void* temp_buf_ = nullptr;
};

}  // namespace infini::ops

#endif
