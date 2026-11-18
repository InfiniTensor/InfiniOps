#ifndef INFINI_OPS_KUNLUN_TRIL_KERNEL_H_
#define INFINI_OPS_KUNLUN_TRIL_KERNEL_H_

#include <stdexcept>

#include "base/tril.h"
#include "native/kunlun/launch2_declarations.h"
#include "native/kunlun/runtime_.h"
#include "native/kunlun/triangular_detail.h"

namespace infini::ops {

template <>
class Operator<Tril, Device::Type::kKunlun> : public Tril {
 public:
  using Tril::Tril;

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, const int64_t diagonal,
                  Tensor out) const override {
    detail::KunlunLaunchTriangular(
        false, input, out, diagonal,
        input.ndim() >= 2 ? input.shape()[input.ndim() - 2] : 1,
        stream_ ? stream_ : nullptr);
  }
};

}  // namespace infini::ops

#endif
