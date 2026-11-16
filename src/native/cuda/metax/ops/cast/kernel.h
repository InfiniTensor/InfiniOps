#ifndef INFINI_OPS_METAX_CAST_KERNEL_H_
#define INFINI_OPS_METAX_CAST_KERNEL_H_

#include <mcr/mc_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "base/cast.h"
#include "native/cuda/metax/ops/cast/kernel.cuh"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kMetax> : public Cast {
 public:
  static bool supports(const Tensor input, const Tensor out) {
    const auto fp32 = DataType::kFloat32, bf16 = DataType::kBFloat16;
    const bool pair = (input.dtype() == fp32 && out.dtype() == bf16) ||
                      (input.dtype() == bf16 && out.dtype() == fp32);
    if (!pair || input.device().type() != Device::Type::kMetax ||
        input.device() != out.device() || input.numel() != out.numel())
      return false;
    size_t input_bytes = 0, output_bytes = 0;
    if (!StorageSpan(input, false, &input_bytes) ||
        !StorageSpan(out, true, &output_bytes))
      return false;
    int device = -1;
    if (mcGetDevice(&device) != mcSuccess || device != input.device().index())
      return false;
    const auto left = reinterpret_cast<uintptr_t>(input.data());
    const auto right = reinterpret_cast<uintptr_t>(out.data());
    return !input_bytes || !output_bytes ||
           (left <= right ? right - left >= input_bytes
                          : left - right >= output_bytes);
  }

  Operator(const Tensor input, Tensor out)
      : Cast(input, out), device_(input.device()) {
    assert(supports(input, out) && "unsupported `Cast` metadata or overlap");
    input_layout_ = MakeLayout(input);
    output_layout_ = MakeLayout(out);
  }

  void operator()(const Tensor input, Tensor out) const override {
    assert(input.dtype() == input_dtype_ && out.dtype() == out_dtype_ &&
           input.shape() == input_shape_ && out.shape() == out_shape_ &&
           input.strides() == input_strides_ && out.strides() == out_strides_ &&
           input.device() == device_ && supports(input, out));
    if (output_size_ == 0) return;
    const unsigned blocks = static_cast<unsigned>(
        std::min<size_t>((output_size_ - 1) / 256 + 1, 65535));
    const auto stream = static_cast<mcStream_t>(stream_);
    if (out_dtype_ == DataType::kFloat32) {
      metax_cast_detail::CastKernel<true><<<blocks, 256, 0, stream>>>(
          input.data(), output_size_, input_layout_, output_layout_,
          out.data());
    } else {
      metax_cast_detail::CastKernel<false><<<blocks, 256, 0, stream>>>(
          input.data(), output_size_, input_layout_, output_layout_,
          out.data());
    }
  }

 private:
  static bool StorageSpan(const Tensor value, bool output, size_t* bytes) {
    if (value.ndim() > metax_cast_detail::kMaxRank ||
        value.ndim() != value.strides().size())
      return false;
    const size_t width = value.dtype() == DataType::kFloat32 ? 4 : 2;
    const size_t limit = std::numeric_limits<ptrdiff_t>::max() / width;
    size_t count = 1, span = 1;
    bool empty = false;
    for (size_t axis = 0; axis < value.ndim(); ++axis) {
      if (value.stride(axis) < 0) return false;
      empty = empty || value.size(axis) == 0;
    }
    if (empty) {
      *bytes = 0;
      return true;
    }
    for (size_t axis = 0; axis < value.ndim(); ++axis) {
      const size_t size = value.size(axis), stride = value.stride(axis);
      if (size > limit / count ||
          (size > 1 && stride > (limit - span) / (size - 1)))
        return false;
      count *= size;
      span += (size - 1) * stride;
    }
    if (output) {
      std::vector<size_t> axes(value.ndim());
      std::iota(axes.begin(), axes.end(), 0);
      std::sort(axes.begin(), axes.end(), [&](size_t a, size_t b) {
        return value.stride(a) < value.stride(b);
      });
      size_t occupied = 1;
      for (size_t axis : axes) {
        if (value.size(axis) <= 1) continue;
        if (static_cast<size_t>(value.stride(axis)) < occupied) return false;
        occupied += (value.size(axis) - 1) * value.stride(axis);
      }
    }
    *bytes = span * width;
    const auto address = reinterpret_cast<uintptr_t>(value.data());
    return address && address % width == 0 &&
           address <= std::numeric_limits<uintptr_t>::max() - *bytes;
  }

  static metax_cast_detail::Layout MakeLayout(const Tensor value) {
    metax_cast_detail::Layout layout;
    layout.rank = value.ndim();
    for (size_t axis = 0; axis < value.ndim(); ++axis) {
      layout.shape[axis] = value.size(axis);
      layout.strides[axis] = value.stride(axis);
    }
    return layout;
  }

  Device device_;

  metax_cast_detail::Layout input_layout_;

  metax_cast_detail::Layout output_layout_;
};

}  // namespace infini::ops

#endif
