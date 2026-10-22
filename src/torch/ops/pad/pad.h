#ifndef INFINI_OPS_TORCH_PAD_H_
#define INFINI_OPS_TORCH_PAD_H_

#include <limits>

#include "base/pad.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<Pad, kDev, 1> : public Pad {
 public:
  static bool supports(const Tensor input, const std::vector<int64_t> pad,
                       const std::string mode, const std::optional<double>,
                       Tensor out) {
    if (mode != "constant" || input.device().type() != kDev ||
        input.device() != out.device() || input.dtype() != out.dtype() ||
        input.ndim() != out.ndim() || pad.size() % 2 ||
        pad.size() / 2 > input.ndim())
      return false;
    for (size_t d = 0; d < input.ndim(); ++d) {
      const auto pair = input.ndim() - 1 - d;
      const auto left = pair < pad.size() / 2 ? pad[2 * pair] : 0;
      const auto right = pair < pad.size() / 2 ? pad[2 * pair + 1] : 0;
      constexpr auto max = std::numeric_limits<int64_t>::max();
      constexpr auto min = std::numeric_limits<int64_t>::min();
      if (input.shape()[d] > static_cast<size_t>(max)) return false;
      auto size = static_cast<int64_t>(input.shape()[d]);
      for (const auto delta : {left, right}) {
        if ((delta > 0 && size > max - delta) ||
            (delta < 0 && size < min - delta))
          return false;
        size += delta;
      }
      if (size < 0 || static_cast<size_t>(size) != out.shape()[d]) return false;
    }
    return true;
  }

  Operator(const Tensor input, const std::vector<int64_t> pad,
           const std::string mode, const std::optional<double> value,
           Tensor out)
      : Pad(input, pad, mode, value, out) {}

  void operator()(const Tensor input, const std::vector<int64_t> pad,
                  const std::string mode, const std::optional<double> value,
                  Tensor out) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_PAD_H_
