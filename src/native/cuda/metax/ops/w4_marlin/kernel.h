#ifndef INFINI_OPS_METAX_W4_MARLIN_KERNEL_H_
#define INFINI_OPS_METAX_W4_MARLIN_KERNEL_H_

#include <stdexcept>

#include "base/w4_marlin_gemm.h"
#include "base/w4_marlin_parameters.h"
#include "base/w4_marlin_repack.h"

namespace infini::ops {

// W4 Marlin is NVIDIA-CUTLASS-specific and has no MetaX
// implementation. The operators are registered so that models
// referencing them link; calling them on MetaX throws.
template <>
class Operator<W4MarlinGemm, Device::Type::kMetax, 0> : public W4MarlinGemm {
 public:
  using W4MarlinGemm::W4MarlinGemm;
  using W4MarlinGemm::operator();

  bool capture_safe() const override { return false; }

  void operator()(const Tensor, const Tensor, const Tensor, const Tensor,
                  const Tensor, const Tensor, const Tensor, const int64_t,
                  const int64_t, const bool, const float,
                  Tensor) const override {
    throw std::runtime_error("W4MarlinGemm is not supported on MetaX");
  }
};

template <>
class Operator<W4MarlinRepack, Device::Type::kMetax, 0>
    : public W4MarlinRepack {
 public:
  using W4MarlinRepack::W4MarlinRepack;
  using W4MarlinRepack::operator();

  bool capture_safe() const override { return false; }

  void operator()(const Tensor, const Tensor, const int64_t, const int64_t,
                  const int64_t, Tensor) const override {
    throw std::runtime_error("W4MarlinRepack is not supported on MetaX");
  }
};

template <>
class Operator<W4MarlinParameters, Device::Type::kMetax, 0>
    : public W4MarlinParameters {
 public:
  using W4MarlinParameters::W4MarlinParameters;
  using W4MarlinParameters::operator();

  bool capture_safe() const override { return false; }

  void operator()(const Tensor, const Tensor, const int64_t, const int64_t,
                  const int64_t, const int64_t, Tensor, Tensor) const override {
    throw std::runtime_error("W4MarlinParameters is not supported on MetaX");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_W4_MARLIN_KERNEL_H_
