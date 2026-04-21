#ifndef INFINI_OPS_BASE_ADD_RMS_NORM_H_
#define INFINI_OPS_BASE_ADD_RMS_NORM_H_

#include <cstddef>
#include <vector>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

// Fused add + `RmsNorm`:
//   `residual_out = input + other`
//   `out = rms_norm(residual_out, weight, eps)`
//
// `residual_out` is produced so downstream layers can keep accumulating a
// running residual without redoing the add.
class AddRmsNorm : public Operator<AddRmsNorm> {
 public:
  // TODO: Make `eps` an `std::optional<float>` with a PyTorch-aligned default.
  // Also consider the same change for `RmsNorm`.
  AddRmsNorm(const Tensor input, const Tensor other, const Tensor weight,
             float eps, Tensor out, Tensor residual_out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        other_strides_{other.strides()},
        weight_shape_{weight.shape()},
        weight_strides_{weight.strides()},
        out_strides_{out.strides()},
        residual_out_strides_{residual_out.strides()},
        input_type_{input.dtype()},
        other_type_{other.dtype()},
        weight_type_{weight.dtype()},
        out_type_{out.dtype()},
        residual_out_type_{residual_out.dtype()},
        eps_{eps} {
    assert(input.dtype() == other.dtype());
    assert(input.dtype() == out.dtype());
    assert(input.dtype() == residual_out.dtype());
  }

  virtual void operator()(const Tensor input, const Tensor other,
                          const Tensor weight, float eps, Tensor out,
                          Tensor residual_out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides other_strides_;

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  Tensor::Strides out_strides_;

  Tensor::Strides residual_out_strides_;

  const DataType input_type_;

  const DataType other_type_;

  const DataType weight_type_;

  const DataType out_type_;

  const DataType residual_out_type_;

  float eps_{1e-6f};
};

}  // namespace infini::ops

#endif
