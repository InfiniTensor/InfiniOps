#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor.h"

namespace infini::ops::cpu_cast_detail {
inline std::size_t Span(const Tensor value) {
  if (value.shape().size() != value.strides().size())
    throw std::invalid_argument("CPU Cast rank/stride mismatch");
  if (!value.numel()) return 0;
  const auto limit = std::numeric_limits<std::size_t>::max();
  std::size_t last = 0;
  for (std::size_t i = 0; i < value.ndim(); ++i) {
    if (value.stride(i) < 0)
      throw std::invalid_argument("CPU Cast negative stride");
    const auto extent = value.size(i) - 1;
    const auto stride = static_cast<std::size_t>(value.stride(i));
    if (extent && stride > (limit - last) / extent)
      throw std::invalid_argument("CPU Cast span overflow");
    last += extent * stride;
  }
  const auto width = value.element_size();
  if (!width || last == limit || last + 1 > limit / width)
    throw std::invalid_argument("CPU Cast byte span overflow");
  const auto bytes = (last + 1) * width;
  const auto pointer = reinterpret_cast<std::uintptr_t>(value.data());
  if (!pointer || pointer % width ||
      bytes > std::numeric_limits<std::uintptr_t>::max() - pointer)
    throw std::invalid_argument("CPU Cast invalid storage");
  return bytes;
}
inline void Validate(const Tensor input, const Tensor out) {
  if (input.device().type() != Device::Type::kCpu ||
      input.device() != out.device() || input.numel() != out.numel())
    throw std::invalid_argument(
        "CPU Cast requires matching CPU element counts");
  const auto ib = Span(input), ob = Span(out);
  if (!ib || !ob) return;
  std::vector<std::pair<std::size_t, std::size_t>> dimensions;
  for (std::size_t i = 0; i < out.ndim(); ++i)
    if (out.size(i) > 1) dimensions.emplace_back(out.stride(i), out.size(i));
  std::sort(dimensions.begin(), dimensions.end());
  std::size_t span = 1;
  for (const auto& [stride, size] : dimensions) {
    if (stride < span)
      throw std::invalid_argument("CPU Cast output elements overlap");
    span += (size - 1) * stride;  // Span already checked the complete sum.
  }
  const auto ip = reinterpret_cast<std::uintptr_t>(input.data());
  const auto op = reinterpret_cast<std::uintptr_t>(out.data());
  const bool exact = ip == op && input.dtype() == out.dtype() &&
                     input.shape() == out.shape() &&
                     input.strides() == out.strides();
  if (!exact && (ip <= op ? op - ip < ib : ip - op < ob))
    throw std::invalid_argument("CPU Cast input and output storage overlap");
}
inline Tensor Checked(const Tensor input, const Tensor out) {
  Validate(input, out);
  return input;
}
}  // namespace infini::ops::cpu_cast_detail
