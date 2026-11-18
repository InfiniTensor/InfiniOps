#ifndef INFINI_OPS_KUNLUN_ARGMAX_KERNEL_H_
#define INFINI_OPS_KUNLUN_ARGMAX_KERNEL_H_

#include <stdexcept>

#include "base/argmax.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/launch2_declarations.h"
#include "native/kunlun/runtime_.h"

namespace infini::ops {

// Kunlun argmax: full-row reduction (dim = nullopt) or last-dim reduction on
// rank-2 inputs; output int64; ties resolve to the smallest index.
template <>
class Operator<Argmax, Device::Type::kKunlun> : public Argmax {
 public:
  using Argmax::Argmax;

  bool capture_safe() const override { return true; }

  static bool supports(const Tensor input, const std::optional<int64_t> dim,
                       const bool keepdim, const Tensor out) {
    if (!input.IsContiguous() || !out.IsContiguous() || keepdim) return false;
    if (out.dtype() != DataType::kInt64) return false;
    if (input.dtype() != DataType::kFloat32 &&
        input.dtype() != DataType::kFloat16 &&
        input.dtype() != DataType::kBFloat16)
      return false;
    if (input.ndim() == 1) {
      return !dim.has_value() && out.ndim() == 0;
    }
    if (input.ndim() == 2) {
      return (!dim.has_value() || dim == 1) && out.ndim() == 1 &&
             out.shape()[0] == input.shape()[0];
    }
    return false;
  }

  void operator()(const Tensor input, const std::optional<int64_t> dim,
                  const bool keepdim, Tensor out) const override {
    if (!supports(input, dim, keepdim, out)) {
      throw std::runtime_error(
          "kunlun argmax: unsupported shape/dtype (1D-2D contiguous only)");
    }
    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    const int tag = detail::KunlunDtypeTag(input_type_);
    const unsigned long long rows =
        input.ndim() == 2 ? (unsigned long long)input.shape()[0] : 1ULL;
    const int width = (int)input.shape().back();
    kunlun_launch::LaunchArgmax(tag, out.data(), input.data(),
                                (unsigned int)rows, width, (long long)width,
                                stream);
  }
};

}  // namespace infini::ops

#endif
