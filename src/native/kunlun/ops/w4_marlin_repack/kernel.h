#ifndef INFINI_OPS_KUNLUN_W4_MARLIN_REPACK_KERNEL_H_
#define INFINI_OPS_KUNLUN_W4_MARLIN_REPACK_KERNEL_H_

#include <stdexcept>

#include "base/w4_marlin_repack.h"

namespace infini::ops {

template <>
class Operator<W4MarlinRepack, Device::Type::kKunlun> : public W4MarlinRepack {
 public:
  using W4MarlinRepack::W4MarlinRepack;

  [[noreturn]] void operator()(const Tensor qweight, const Tensor perm,
                               const int64_t scheme, const int64_t size_k,
                               const int64_t size_n,
                               Tensor out) const override {
    throw std::runtime_error("W4MarlinRepack is not implemented on Kunlun");
  }
};

}  // namespace infini::ops

#endif
