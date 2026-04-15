#ifndef INFINI_OPS_ASCEND_SWIGLU_KERNEL_H_
#define INFINI_OPS_ASCEND_SWIGLU_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_mul.h"
#include "aclnn_silu.h"
#include "ascend/common.h"
#include "ascend/swiglu/registry.h"
#include "ascend/workspace_pool_.h"
#include "base/swiglu.h"
#include "data_type.h"
#include "operator.h"

namespace infini::ops {

// Implements SwiGLU as two ACLNN calls: silu(gate) into a temp buffer,
// then elementwise mul(input, temp) into out.
// `aclnnSiluMul` was not used because it fuses silu_AND_mul on the same
// tensor (x * silu(x)), whereas SwiGLU requires input * silu(gate) —
// two distinct inputs.
template <>
class Operator<Swiglu, Device::Type::kAscend, 0> : public Swiglu {
 public:
  Operator(const Tensor input, const Tensor gate, Tensor out)
      : Swiglu(input, gate, out),
        in_cache_(input),
        gate_cache_(gate),
        out_cache_(out) {
    temp_size_ = input.numel() * kDataTypeToSize.at(input.dtype());

    // Build temp cache from gate geometry (contiguous, same shape/dtype).
    // No data pointer yet — will be set on first `get()` call.
    Tensor temp_t{nullptr, gate.shape(), gate.dtype(), gate.device()};
    temp_cache_ = ascend::AclTensorCache(temp_t);
  }

  ~Operator() {
    if (silu_exec_) aclDestroyAclOpExecutor(silu_exec_);
    if (mul_exec_) aclDestroyAclOpExecutor(mul_exec_);
  }

  void operator()(const Tensor input, const Tensor gate,
                  Tensor out) const override {
    auto t_in = in_cache_.get(const_cast<void*>(input.data()));
    auto t_gate = gate_cache_.get(const_cast<void*>(gate.data()));
    auto t_out = out_cache_.get(out.data());
    auto stream = static_cast<aclrtStream>(stream_);

    // Obtain shared temp buffer from pool.
    auto& temp = ascend::workspacePool().ensure(stream, temp_size_, "temp");
    auto t_temp = temp_cache_.get(temp.buf);

    // Step 1: silu(gate) -> temp.
    if (!silu_exec_) {
      aclnnSiluGetWorkspaceSize(t_gate, t_temp, &silu_ws_, &silu_exec_);
      aclSetAclOpExecutorRepeatable(silu_exec_);
    } else {
      aclSetInputTensorAddr(silu_exec_, 0, t_gate,
                            const_cast<void*>(gate.data()));
      aclSetOutputTensorAddr(silu_exec_, 0, t_temp, temp.buf);
    }
    auto& silu_arena = ascend::workspacePool().ensure(stream, silu_ws_);
    aclnnSilu(silu_arena.buf, silu_ws_, silu_exec_, stream);

    // Step 2: mul(input, temp) -> out.
    if (!mul_exec_) {
      aclnnMulGetWorkspaceSize(t_in, t_temp, t_out, &mul_ws_, &mul_exec_);
      aclSetAclOpExecutorRepeatable(mul_exec_);
    } else {
      aclSetInputTensorAddr(mul_exec_, 0, t_in,
                            const_cast<void*>(input.data()));
      aclSetInputTensorAddr(mul_exec_, 1, t_temp, temp.buf);
      aclSetOutputTensorAddr(mul_exec_, 0, t_out, out.data());
    }
    auto& mul_arena = ascend::workspacePool().ensure(stream, mul_ws_);
    aclnnMul(mul_arena.buf, mul_ws_, mul_exec_, stream);
  }

 private:
  mutable ascend::AclTensorCache in_cache_;

  mutable ascend::AclTensorCache gate_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable ascend::AclTensorCache temp_cache_;

  uint64_t temp_size_ = 0;

  mutable aclOpExecutor* silu_exec_ = nullptr;

  mutable uint64_t silu_ws_ = 0;

  mutable aclOpExecutor* mul_exec_ = nullptr;

  mutable uint64_t mul_ws_ = 0;
};

}  // namespace infini::ops

#include "ascend/swiglu/kernel_fused.h"

#endif
