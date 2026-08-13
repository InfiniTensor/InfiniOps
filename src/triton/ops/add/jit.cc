#include "triton/ops/add/jit.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "triton/jit/jit.h"

namespace infini::ops {

namespace {

using triton::jit::Config;
using triton::jit::Grid;

const Config& DefaultConfig() {
  static const Config config{4u, 3u, {{"BLOCK_SIZE", 1024}}};
  return config;
}

std::optional<Grid> MakeGrid(const Config& config, Tensor::Size num_elements) {
  const auto block_size = config.FindConstexpr("BLOCK_SIZE");
  assert(block_size.has_value() && *block_size > 0 &&
         "Triton JIT `Add` requires a positive `BLOCK_SIZE`.");
  if (!block_size.has_value() || *block_size <= 0) return std::nullopt;

  const auto block_size_value = static_cast<Tensor::Size>(*block_size);
  const auto block_count =
      num_elements / block_size_value +
      static_cast<Tensor::Size>(num_elements % block_size_value != 0);
  assert(block_count <= std::numeric_limits<unsigned>::max() &&
         "Triton JIT `Add` grid does not fit in an unsigned integer.");
  if (block_count > std::numeric_limits<unsigned>::max()) return std::nullopt;

  return Grid{static_cast<unsigned>(block_count)};
}

}  // namespace

void Operator<Add, Device::Type::kNvidia, 10>::operator()(
    const Tensor input, const Tensor other, const double alpha,
    Tensor out) const {
  const Device device = out.device();
  const bool devices_match = input.device().type() == device.type() &&
                             input.device().index() == device.index() &&
                             other.device().type() == device.type() &&
                             other.device().index() == device.index();
  assert(devices_match &&
         "Triton JIT `Add` inputs and output must be on the same device.");
  if (!devices_match) return;

  const bool shapes_match =
      input.shape() == other.shape() && input.shape() == out.shape();
  assert(shapes_match &&
         "Triton JIT `Add` requires input and output shapes to match.");
  if (!shapes_match) return;

  const bool tensors_are_contiguous = input.IsContiguous() &&
                                      other.IsContiguous() &&
                                      out.IsContiguous();
  assert(tensors_are_contiguous &&
         "Triton JIT `Add` requires contiguous tensors.");
  if (!tensors_are_contiguous) return;

  assert(alpha == 1.0 && "Triton JIT `Add` only supports alpha equal to one.");
  if (alpha != 1.0) return;

  const Tensor::Size num_elements = out.numel();
  if (num_elements == 0) return;

  const auto& default_config = DefaultConfig();
  triton::jit::Launch<Device::Type::kNvidia>(
      "add", device.index(), this->stream_, this->config_ptr_.get(),
      default_config,
      {{"num_elements", static_cast<std::uint64_t>(num_elements)}},
      [num_elements](const Config& config) {
        return MakeGrid(config, num_elements);
      },
      input, other, out, num_elements);
}

}  // namespace infini::ops
