#ifndef INFINI_OPS_ASCEND_PAD_KERNEL_H_
#define INFINI_OPS_ASCEND_PAD_KERNEL_H_

#include <limits>
#include "base/pad.h"
#include "native/ascend/ops/copy/kernel.h"
#include "native/ascend/ops/fill/kernel.h"
#include "native/cuda/binary_view.h"

namespace infini::ops {
template <>
class Operator<Pad, Device::Type::kAscend> : public Pad {
 public:
  static bool supports(const Tensor input, const std::vector<int64_t> pad,
                       const std::string mode, const std::optional<double>,
                       Tensor out) {
    if (mode != "constant" || input.device().type() != Device::Type::kAscend ||
        input.device() != out.device() || input.dtype() != out.dtype() ||
        input.ndim() != out.ndim() || input.ndim() > 8 || pad.size() % 2 ||
        pad.size() / 2 > input.ndim() || out.HasBroadcastDim() || out.numel() == 0) return false;
    if (input.dtype() != DataType::kFloat32 && input.dtype() != DataType::kFloat16 &&
        input.dtype() != DataType::kBFloat16) return false;
    std::size_t input_span = 0, output_span = 0;
    if (!cuda_binary_view::SpanBytes(input, &input_span) ||
        !cuda_binary_view::SpanBytes(out, &output_span) ||
        !cuda_binary_view::NonOverlappingOutput(out) ||
        cuda_binary_view::Overlaps(input, input_span, out, output_span))
      return false;
    constexpr auto max = std::numeric_limits<int64_t>::max();
    for (size_t d = 0; d < input.ndim(); ++d) {
      if (input.shape()[d] > static_cast<size_t>(max) || input.strides()[d] < 0 ||
          out.strides()[d] <= 0) return false;
      auto size = static_cast<int64_t>(input.shape()[d]);
      const auto pair = input.ndim() - 1 - d;
      const auto left = pair < pad.size() / 2 ? pad[2 * pair] : 0;
      const auto right = pair < pad.size() / 2 ? pad[2 * pair + 1] : 0;
      if (left < 0 || right < 0 || size > max - left) return false;
      size += left;
      if (size > max - right) return false;
      size += right;
      if (static_cast<size_t>(size) != out.shape()[d]) return false;
    }
    return true;
  }

  Operator(const Tensor input, const std::vector<int64_t> pad,
           const std::string mode, const std::optional<double> value, Tensor out)
      : Pad(input, pad, mode, value, out),
        fill_(out, value.value_or(0.0), out),
        copy_(input, false, Interior(input, pad, out)) {}

  void operator()(const Tensor input, const std::vector<int64_t> pad,
                  const std::string, const std::optional<double> value,
                  Tensor out) const override {
    fill_.set_stream(stream_);
    fill_(out, value.value_or(0.0), out);
    if (!input.numel()) return;
    copy_.set_stream(stream_);
    copy_(input, false, Interior(input, pad, out));
  }

 private:
  static Tensor Interior(const Tensor input, const std::vector<int64_t>& pad,
                         Tensor out) {
    std::size_t offset = 0;
    for (std::size_t i = 0; i < pad.size() / 2; ++i)
      offset += static_cast<std::size_t>(pad[2 * i]) *
                static_cast<std::size_t>(out.strides()[out.ndim() - 1 - i]);
    return Tensor{static_cast<char*>(out.data()) + offset * out.element_size(),
                  input.shape(), input.dtype(), out.device(), out.strides()};
  }

  mutable Operator<Fill, Device::Type::kAscend> fill_;
  mutable Operator<Copy, Device::Type::kAscend> copy_;
};
}  // namespace infini::ops
#endif
