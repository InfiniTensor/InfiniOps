#ifndef INFINI_OPS_CPU_CAST_CAST_H_
#define INFINI_OPS_CPU_CAST_CAST_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "base/cast.h"
#include "common/generic_utils.h"
#include "native/cpu/caster_.h"
#include "native/cpu/dense_view.h"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kCpu> : public Cast {
 public:
  static bool supports(const Tensor input, Tensor out) {
    if (input.device().type() != Device::Type::kCpu ||
        input.device() != out.device() || input.shape() != out.shape() ||
        !detail::ListContains(input.dtype(), AllTypes{}) ||
        !detail::ListContains(out.dtype(), AllTypes{}))
      return false;
    std::size_t input_bytes = 0, out_bytes = 0;
    if (!StorageBytes(input, &input_bytes) || !StorageBytes(out, &out_bytes) ||
        !cpu_dense_view::ValidAddress(input, input_bytes) ||
        !cpu_dense_view::ValidAddress(out, out_bytes))
      return false;
    if (!out_bytes || SameView(input, out)) return true;
    if (!NonOverlappingOutput(out)) return false;
    const auto ip = reinterpret_cast<std::uintptr_t>(input.data());
    const auto op = reinterpret_cast<std::uintptr_t>(out.data());
    return ip >= op + out_bytes || op >= ip + input_bytes;
  }

  Operator(const Tensor input, Tensor out)
      : Cast{Checked(input, out), out}, device_{input.device()} {}

  void operator()(const Tensor input, Tensor out) const override {
    // Cached descriptors do not include data pointers, so recheck aliasing.
    if (input.dtype() != input_dtype_ || out.dtype() != out_dtype_ ||
        input.shape() != input_shape_ || out.shape() != out_shape_ ||
        input.strides() != input_strides_ || out.strides() != out_strides_ ||
        input.device() != device_ || !supports(input, out))
      throw std::invalid_argument(
          "CPU `Cast`: Call metadata or alias changed after `Make`.");
    if (output_size_ == 0 || SameView(input, out)) return;
    DispatchFunc<Device::Type::kCpu, AllTypes, AllTypes>(
        {input_dtype_, out_dtype_},
        [&](auto in_tag, auto out_tag) {
          using InT = typename decltype(in_tag)::type;
          using OutT = typename decltype(out_tag)::type;
          Compute<InT, OutT>(input, out);
        },
        "`Operator<Cast, Device::Type::kCpu>::operator()`");
  }

 private:
  static bool SameView(const Tensor input, const Tensor out) {
    return input.dtype() == out.dtype() && input.data() == out.data() &&
           input.shape() == out.shape() && input.strides() == out.strides();
  }

  static Tensor Checked(const Tensor input, Tensor out) {
    if (!supports(input, out))
      throw std::invalid_argument(
          "CPU `Cast`: Unsupported dtype/device/layout or alias.");
    return input;
  }

  static bool StorageBytes(const Tensor tensor, std::size_t* bytes) {
    if (tensor.shape().size() != tensor.strides().size()) return false;
    const auto limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
        tensor.element_size();
    bool empty = false;
    for (std::size_t axis = 0; axis < tensor.ndim(); ++axis) {
      if (tensor.stride(axis) < 0 || tensor.size(axis) > limit) return false;
      empty = empty || tensor.size(axis) == 0;
    }
    if (empty) {
      *bytes = 0;
      return true;
    }
    std::size_t count = 1, span = 1;
    for (std::size_t axis = 0; axis < tensor.ndim(); ++axis) {
      const auto size = tensor.size(axis);
      const auto stride = static_cast<std::size_t>(tensor.stride(axis));
      if (size > limit / count ||
          (size > 1 && stride > (limit - span) / (size - 1)))
        return false;
      count *= size;
      span += (size - 1) * stride;
    }
    *bytes = span * tensor.element_size();
    return true;
  }

  static bool NonOverlappingOutput(const Tensor out) {
    std::vector<std::pair<std::size_t, std::size_t>> axes;
    for (std::size_t axis = 0; axis < out.ndim(); ++axis)
      if (out.size(axis) > 1)
        axes.emplace_back(out.stride(axis), out.size(axis));
    std::sort(axes.begin(), axes.end());
    std::size_t span = 1;
    for (const auto& [stride, size] : axes) {
      if (stride < span) return false;
      // `StorageBytes` already checked the same extent for overflow.
      span += (size - 1) * stride;
    }
    return true;
  }

  template <typename InT, typename OutT>
  void Compute(const Tensor input, Tensor out) const {
    const auto* in_ptr = static_cast<const InT*>(input.data());
    auto* out_ptr = static_cast<OutT*>(out.data());

    auto get_idx = [&](Tensor::Size i, bool is_contig, const auto* shape,
                       const auto* strides) {
      return is_contig ? i : utils::IndexToOffset(i, ndim_, shape, strides);
    };

#pragma omp parallel for
    for (Tensor::Size i = 0; i < output_size_; ++i) {
      auto in_idx = get_idx(i, is_input_contiguous_, input_shape_.data(),
                            input_strides_.data());
      auto out_idx = get_idx(i, is_out_contiguous_, out_shape_.data(),
                             out_strides_.data());

      out_ptr[out_idx] =
          Caster<Device::Type::kCpu>::template Cast<OutT>(in_ptr[in_idx]);
    }
  }

  Device device_;
};

}  // namespace infini::ops

#endif
