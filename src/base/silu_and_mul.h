#ifndef INFINI_OPS_BASE_SILU_AND_MUL_H_
#define INFINI_OPS_BASE_SILU_AND_MUL_H_

#include "operator.h"

namespace infini::ops {

// vLLM-aligned SiLU-gated linear unit.
//
// Mirrors `vllm._C.silu_and_mul(result, input)` semantics:
// splits `input` into halves along `dim` and computes
// `silu(first_half) * second_half`.  vLLM hardcodes `dim = -1`; this base
// retains `dim` as a parameter for generality but defaults to `-1` to match
// the PyTorch convention for feature-dimension operations (e.g. `softmax`,
// `log_softmax`).
class SiluAndMul : public Operator<SiluAndMul> {
 public:
  SiluAndMul(const Tensor input, int64_t dim, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        input_dtype_{input.dtype()},
        out_dtype_{out.dtype()},
        dim_{dim},
        ndim_{input.ndim()},
        is_input_contiguous_{input.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(input_dtype_ == out_dtype_ &&
           "`SiluAndMul`: `input` and `out` must have the same dtype.");
  }

  // Convenience overload: `dim` defaults to `-1`, matching
  // `torch.nn.functional.glu(input, dim=-1)` and vLLM's hardcoded last-dim
  // behavior.
  SiluAndMul(const Tensor input, Tensor out) : SiluAndMul{input, -1, out} {}

  virtual void operator()(const Tensor input, int64_t dim, Tensor out) const = 0;

  virtual void operator()(const Tensor input, Tensor out) const {
    return operator()(input, -1, out);
  }

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  const DataType input_dtype_;

  const DataType out_dtype_;

  int64_t dim_;

  Tensor::Size ndim_;

  bool is_input_contiguous_;

  bool is_out_contiguous_;
};

}  // namespace infini::ops

#endif
