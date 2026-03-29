#ifndef INFINI_OPS_ASCEND_SWIGLU_KERNEL_H_
#define INFINI_OPS_ASCEND_SWIGLU_KERNEL_H_

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_mul.h"
#include "aclnn_silu.h"
#include "ascend/device.h"
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

  ~Operator() {
    aclrtFree(temp_buf_);
    if (silu_ws_size_ > 0) aclrtFree(silu_ws_);
    if (mul_ws_size_ > 0) aclrtFree(mul_ws_);
  }

  void operator()(const Tensor input, const Tensor gate,
                  Tensor out) const override {
    // temp_buf_ is a contiguous scratch buffer; give it contiguous strides.
    Tensor temp_t{temp_buf_, gate.shape(), gate.dtype(), gate.device()};

    auto t_in   = ascend::buildAclTensor(input);
    auto t_gate = ascend::buildAclTensor(gate);
    auto t_out  = ascend::buildAclTensor(out);
    auto t_temp = ascend::buildAclTensor(temp_t);

    uint64_t ws_needed = 0;
    aclOpExecutor* exec = nullptr;
    auto stream = static_cast<aclrtStream>(stream_);

    // Step 1: silu(gate) -> temp.  SwiGLU = input * silu(gate).
    aclnnSiluGetWorkspaceSize(t_gate, t_temp, &ws_needed, &exec);
    ascend::ensureWorkspace(silu_ws_, silu_ws_size_, ws_needed, stream);
    aclnnSilu(silu_ws_, silu_ws_size_, exec, stream);

    // Step 2: mul(input, temp) -> out.
    ws_needed = 0;
    exec = nullptr;
    aclnnMulGetWorkspaceSize(t_in, t_temp, t_out, &ws_needed, &exec);
    ascend::ensureWorkspace(mul_ws_, mul_ws_size_, ws_needed, stream);
    aclnnMul(mul_ws_, mul_ws_size_, exec, stream);

    aclDestroyTensor(t_in);
    aclDestroyTensor(t_gate);
    aclDestroyTensor(t_out);
    aclDestroyTensor(t_temp);
  }

 private:
  void*            temp_buf_     = nullptr;
  mutable void*    silu_ws_      = nullptr;
  mutable uint64_t silu_ws_size_ = 0;
  mutable void*    mul_ws_       = nullptr;
  mutable uint64_t mul_ws_size_  = 0;
};

}  // namespace infini::ops

#endif
