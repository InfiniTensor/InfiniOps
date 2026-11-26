#ifndef INFINI_OPS_CUDA_MUL_SCALAR_PROVIDER_H_
#define INFINI_OPS_CUDA_MUL_SCALAR_PROVIDER_H_

#include <optional>
#include <utility>

#include "native/cuda/ops/mul/kernel.h"

namespace infini::ops {

// Shared operator wrapper for the CUDA-family scalar-multiplier
// `Mul(tensor, double, tensor)` providers. The per-platform
// `ScalarMul` class supplies the actual kernel capture and launch;
// this wrapper only wires the scalar overload into the operator:
// the shared `CudaMul` scalar constructor is protected and its
// tensor `operator()` rejects scalar mode, so without it dispatch
// of the scalar overload finds no provider.
template <typename Backend, typename ScalarMul>
class CudaScalarMulProvider : public CudaMul<Backend> {
 public:
  using CudaMul<Backend>::CudaMul;
  using CudaMul<Backend>::operator();

  CudaScalarMulProvider(const Tensor input, const double alpha, Tensor out)
      : CudaMul<Backend>{input, alpha, out},
        scalar_(std::in_place, input, alpha, out) {}

  static bool supports(const Tensor input, const double alpha, Tensor out) {
    return ScalarMul::supports(input, alpha, out);
  }

  void operator()(const Tensor input, const double alpha,
                  Tensor out) const override {
    if (!scalar_)
      throw std::invalid_argument("Mul prepared for tensor operands");
    // `this->` is required: the base `CudaMul<Backend>` is dependent, so
    // unqualified lookup of the `OperatorBase::stream_` member fails at
    // the template definition phase.
    scalar_->Run(input, alpha, out,
                 static_cast<typename Backend::Stream>(
                     this->stream_ ? this->stream_ : nullptr));
  }

  // All dispatched types use kernels and constructor-owned metadata on replay.
  bool capture_safe() const override { return true; }

 private:
  std::optional<ScalarMul> scalar_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_MUL_SCALAR_PROVIDER_H_
