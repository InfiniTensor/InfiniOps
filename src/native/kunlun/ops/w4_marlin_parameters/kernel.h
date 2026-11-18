#ifndef INFINI_OPS_KUNLUN_W4_MARLIN_PARAMETERS_KERNEL_H_
#define INFINI_OPS_KUNLUN_W4_MARLIN_PARAMETERS_KERNEL_H_

#include <stdexcept>

#include "base/w4_marlin_parameters.h"

namespace infini::ops {

// Placeholder: W4 Marlin weight repacking has no Kunlun implementation; the
// class must exist for InfiniLM's quantization adapter to compile. Running
// it throws.
template <>
class Operator<W4MarlinParameters, Device::Type::kKunlun>
    : public W4MarlinParameters {
 public:
  using W4MarlinParameters::W4MarlinParameters;

  [[noreturn]] void operator()(const Tensor scales, const Tensor qzeros,
                               const int64_t scheme, const int64_t size_k,
                               const int64_t size_n, const int64_t group_size,
                               Tensor scales_out,
                               Tensor zeros_out) const override {
    (void)scales;
    (void)qzeros;
    (void)scheme;
    (void)size_k;
    (void)size_n;
    (void)group_size;
    (void)scales_out;
    (void)zeros_out;
    throw std::runtime_error("W4MarlinParameters is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
