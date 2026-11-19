#ifndef INFINI_OPS_METAX_MUL_KERNEL_H_
#define INFINI_OPS_METAX_MUL_KERNEL_H_

#include <optional>
#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/ops/mul/scalar.h"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/mul/kernel.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kMetax>
    : public CudaMul<Runtime<Device::Type::kMetax>> {
 public:
  using CudaMul<Runtime<Device::Type::kMetax>>::CudaMul;
  using CudaMul<Runtime<Device::Type::kMetax>>::operator();

  // The shared `CudaMul` scalar constructor is protected and its
  // scalar `operator()` throws, so the inherited `using` declarations
  // never satisfy the scalar `Mul(tensor, double, tensor)` dispatch.
  // The DeepSeek-V2 MoE router scale needs that overload; serve it
  // with a MetaX scalar kernel.
  Operator(const Tensor input, const double alpha, Tensor out)
      : CudaMul<Runtime<Device::Type::kMetax>>{input, alpha, out},
        scalar_(std::in_place, input, alpha, out) {}

  static bool supports(const Tensor input, const double alpha, Tensor out) {
    return MetaxScalarMul::supports(input, alpha, out);
  }

  void operator()(const Tensor input, const double alpha,
                  Tensor out) const override {
    if (!scalar_)
      throw std::invalid_argument("MetaX Mul prepared for tensor operands");
    scalar_->Run(input, alpha, out,
                 static_cast<Runtime<Device::Type::kMetax>::Stream>(
                     stream_ ? stream_ : nullptr));
  }

 private:
  std::optional<MetaxScalarMul> scalar_;
};

}  // namespace infini::ops

#endif
