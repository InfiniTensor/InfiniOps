#ifndef INFINI_OPS_ILUVATAR_ADD_RELU_KERNEL_H_
#define INFINI_OPS_ILUVATAR_ADD_RELU_KERNEL_H_

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>
#include <limits>

#include "base/internal_add_relu.h"
#include "native/cuda/iluvatar/ops/internal_add_relu/kernel.cuh"

namespace infini::ops {

template <>
class Operator<internal::AddRelu, Device::Type::kIluvatar>
    : public internal::AddRelu {
 public:
  using Base = internal::AddRelu;

  static bool supports(const Tensor input, const Tensor other, double alpha,
                       Tensor out) {
    if (alpha != 1.0 || input.device().type() != Device::Type::kIluvatar ||
        input.device() != other.device() || input.device() != out.device() ||
        input.dtype() != DataType::kFloat32 || other.dtype() != input.dtype() ||
        out.dtype() != input.dtype() || input.shape() != out.shape() ||
        !input.IsContiguous() || !other.IsContiguous() || !out.IsContiguous() ||
        Mode(out, other) < 0)
      return false;
    const auto overlaps = [](const Tensor a, const Tensor b) {
      const auto left = reinterpret_cast<uintptr_t>(a.data());
      const auto right = reinterpret_cast<uintptr_t>(b.data());
      return left <= right ? right - left < a.numel() * sizeof(float)
                           : left - right < b.numel() * sizeof(float);
    };
    return !overlaps(out, input) && !overlaps(out, other);
  }

  Operator(const Tensor input, const Tensor other, double alpha, Tensor out)
      : Base(input, other, alpha, out), mode_(Mode(out, other)) {
    assert(supports(input, other, alpha, out));
  }

  void operator()(const Tensor input, const Tensor other, double alpha,
                  Tensor out) const override {
    assert(supports(input, other, alpha, out) &&
           input.shape() == input_shape_ && other.shape() == other_shape_ &&
           out.shape() == out_shape_);
    const auto count = static_cast<uint32_t>(out.numel());
    const auto blocks = static_cast<unsigned>((uint64_t(count) + 255) / 256);
    iluvatar_add_relu_detail::
        AddReluKernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream_)>>>(
            static_cast<const float*>(input.data()),
            static_cast<const float*>(other.data()), count, mode_,
            static_cast<uint32_t>(out.size(2) * out.size(3)),
            static_cast<uint32_t>(out.size(1)),
            static_cast<float*>(out.data()));
  }

 private:
  static int Mode(const Tensor out, const Tensor other) {
    if (out.ndim() != 4 || other.ndim() == 0 || other.ndim() > 4) return -1;
    uint64_t count = 1;
    for (const auto size : out.shape()) {
      if (size == 0 || size > std::numeric_limits<uint32_t>::max() / count)
        return -1;
      count *= size;
    }
    Tensor::Shape expanded(4, 1);
    for (size_t i = 0; i < other.ndim(); ++i)
      expanded[4 - other.ndim() + i] = other.size(i);
    if (expanded == out.shape()) return 0;
    if (expanded == Tensor::Shape({1, 1, 1, 1})) return 1;
    if (expanded == Tensor::Shape({1, out.size(1), 1, 1})) return 2;
    return -1;
  }

  int mode_;
};

}  // namespace infini::ops

#endif
