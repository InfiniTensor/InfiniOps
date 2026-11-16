#ifndef INFINI_OPS_CUDA_LAYER_NORM_KERNEL_H_
#define INFINI_OPS_CUDA_LAYER_NORM_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "base/layer_norm.h"
#include "data_type.h"
#include "native/cuda/caster.cuh"
#include "native/cuda/ops/layer_norm/kernel.cuh"
#include "native/cuda/runtime_utils.h"
#include "tensor.h"

namespace infini::ops {
namespace cuda_layer_norm_detail {

inline bool SpanBytes(const Tensor tensor, std::size_t* bytes) {
  if (tensor.shape().size() != tensor.strides().size() ||
      !tensor.element_size()) {
    return false;
  }
  for (auto dim : tensor.shape()) {
    if (!dim) {
      *bytes = 0;
      return true;
    }
  }

  std::size_t last = 0;
  for (std::size_t i = 0; i < tensor.ndim(); ++i) {
    if (tensor.strides()[i] < 0) return false;
    const auto stride = static_cast<std::size_t>(tensor.strides()[i]);
    const auto extent = tensor.shape()[i] - 1;
    if (extent &&
        stride > (std::numeric_limits<std::size_t>::max() - last) / extent) {
      return false;
    }
    last += extent * stride;
  }
  if (last == std::numeric_limits<std::size_t>::max() ||
      last + 1 >
          std::numeric_limits<std::size_t>::max() / tensor.element_size()) {
    return false;
  }
  *bytes = (last + 1) * tensor.element_size();
  return !*bytes || tensor.data();
}

inline bool NonOverlappingOutput(const Tensor out) {
  if (out.shape().size() != out.strides().size()) return false;
  if (out.HasBroadcastDim()) return false;
  for (auto dim : out.shape()) {
    if (!dim) return true;
  }

  std::vector<std::pair<std::size_t, std::size_t>> dimensions;
  for (std::size_t i = 0; i < out.ndim(); ++i) {
    if (out.strides()[i] < 0) return false;
    if (out.shape()[i] > 1) {
      dimensions.emplace_back(static_cast<std::size_t>(out.strides()[i]),
                              out.shape()[i]);
    }
  }
  std::sort(dimensions.begin(), dimensions.end());

  std::size_t span = 1;
  for (const auto& [stride, extent] : dimensions) {
    if (stride < span ||
        extent - 1 >
            (std::numeric_limits<std::size_t>::max() - span) / stride) {
      return false;
    }
    span += (extent - 1) * stride;
  }
  return true;
}

inline bool Overlaps(const Tensor a, std::size_t a_bytes, const Tensor b,
                     std::size_t b_bytes) {
  if (!a_bytes || !b_bytes) return false;
  const auto lhs = reinterpret_cast<std::uintptr_t>(a.data());
  const auto rhs = reinterpret_cast<std::uintptr_t>(b.data());
  return lhs <= rhs ? rhs - lhs < a_bytes : lhs - rhs < b_bytes;
}

}  // namespace cuda_layer_norm_detail

template <typename Backend>
class CudaLayerNorm : public LayerNorm {
 public:
  static bool supports(const Tensor input, const Tensor weight,
                       const std::optional<Tensor> bias, float, Tensor out) {
    using namespace cuda_layer_norm_detail;
    const auto rank = input.shape().size();
    if (input.device().type() != Backend::kDeviceType ||
        input.device() != weight.device() || input.device() != out.device() ||
        (bias && input.device() != bias->device()) || rank < 1 || rank > 8 ||
        rank != input.strides().size() || rank != out.strides().size() ||
        out.shape() != input.shape() || weight.shape().size() != 1 ||
        weight.strides().size() != 1 ||
        (bias && (bias->shape().size() != 1 || bias->strides().size() != 1)) ||
        input.shape()[rank - 1] == 0 ||
        weight.shape()[0] != input.shape()[rank - 1] ||
        (bias && bias->shape()[0] != input.shape()[rank - 1]) ||
        input.dtype() != weight.dtype() || input.dtype() != out.dtype() ||
        (bias && bias->dtype() != input.dtype()) ||
        (input.dtype() != DataType::kFloat16 &&
         input.dtype() != DataType::kBFloat16 &&
         input.dtype() != DataType::kFloat32) ||
        !NonOverlappingOutput(out)) {
      return false;
    }

    int current = -1;
    if (Backend::GetDevice(&current) != Backend::kSuccess ||
        current != input.device().index()) {
      return false;
    }

    std::size_t rows = 1;
    for (std::size_t i = 0; i + 1 < rank; ++i) {
      if (input.shape()[i] &&
          rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) /
                     input.shape()[i]) {
        return false;
      }
      rows *= input.shape()[i];
    }
    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        input.shape()[rank - 1] >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        weight.strides()[0] < 0 || (bias && bias->strides()[0] < 0)) {
      return false;
    }

    std::size_t input_bytes = 0;
    std::size_t weight_bytes = 0;
    std::size_t out_bytes = 0;
    std::size_t bias_bytes = 0;
    if (!SpanBytes(input, &input_bytes) || !SpanBytes(weight, &weight_bytes) ||
        !SpanBytes(out, &out_bytes) ||
        (bias && !SpanBytes(*bias, &bias_bytes))) {
      return false;
    }

    const auto width = input.element_size();
    if ((input_bytes &&
         reinterpret_cast<std::uintptr_t>(input.data()) % width) ||
        (weight_bytes &&
         reinterpret_cast<std::uintptr_t>(weight.data()) % width) ||
        (out_bytes && reinterpret_cast<std::uintptr_t>(out.data()) % width) ||
        (bias_bytes &&
         reinterpret_cast<std::uintptr_t>(bias->data()) % width)) {
      return false;
    }

    const bool exact_inplace = input.data() == out.data() &&
                               input.strides() == out.strides() &&
                               NonOverlappingOutput(input);
    return !Overlaps(out, out_bytes, weight, weight_bytes) &&
           (!bias || !Overlaps(out, out_bytes, *bias, bias_bytes)) &&
           (!Overlaps(out, out_bytes, input, input_bytes) || exact_inplace);
  }

  CudaLayerNorm(const Tensor input, const Tensor weight,
                const std::optional<Tensor> bias, float eps, Tensor out)
      : LayerNorm(checked(input, weight, bias, eps, out), weight, bias, eps,
                  out) {
    using namespace cuda_layer_norm_detail;
    SpanBytes(input, &input_bytes_);
    SpanBytes(weight, &weight_bytes_);
    SpanBytes(out, &out_bytes_);
    if (bias) SpanBytes(*bias, &bias_bytes_);
    exact_inplace_layout_ = input_meta_.strides() == out_meta_.strides();

    layout_.rank = static_cast<unsigned>(input.ndim());
    layout_.dim = input.size(-1);
    layout_.weight_stride = static_cast<std::size_t>(weight.stride(0));
    layout_.bias_stride = bias ? static_cast<std::size_t>(bias->stride(0)) : 0;
    for (std::size_t i = 0; i < input.ndim(); ++i) {
      layout_.shape[i] = input.shape()[i];
      layout_.input_strides[i] = static_cast<std::size_t>(input.strides()[i]);
      layout_.out_strides[i] = static_cast<std::size_t>(out.strides()[i]);
      if (i + 1 < input.ndim()) rows_ *= input.shape()[i];
    }
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, const Tensor weight,
                  const std::optional<Tensor> bias, float eps,
                  Tensor out) const override {
    using namespace cuda_layer_norm_detail;
    if (!std::equal_to<Tensor>{}(input, input_meta_) ||
        !std::equal_to<Tensor>{}(weight, weight_meta_) ||
        !std::equal_to<Tensor>{}(out, out_meta_) ||
        bias.has_value() != bias_meta_.has_value() ||
        (bias && !std::equal_to<Tensor>{}(*bias, *bias_meta_)) ||
        bits(eps) != bits(eps_)) {
      throw std::invalid_argument(
          "CUDA LayerNorm call metadata or alias changed after Make");
    }

    int current = -1;
    if (Backend::GetDevice(&current) != Backend::kSuccess ||
        current != device_.index() ||
        Overlaps(out, out_bytes_, weight, weight_bytes_) ||
        (bias && Overlaps(out, out_bytes_, *bias, bias_bytes_)) ||
        (Overlaps(out, out_bytes_, input, input_bytes_) &&
         !(input.data() == out.data() && exact_inplace_layout_))) {
      throw std::invalid_argument(
          "CUDA LayerNorm current device or alias changed after Make");
    }
    if (!rows_) return;

    auto stream = static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    switch (input_meta_.dtype()) {
      case DataType::kFloat16:
        LaunchKernel<TypeMapType<Backend::kDeviceType, DataType::kFloat16>>(
            input, weight, bias, out, stream);
        break;
      case DataType::kBFloat16:
        LaunchKernel<TypeMapType<Backend::kDeviceType, DataType::kBFloat16>>(
            input, weight, bias, out, stream);
        break;
      case DataType::kFloat32:
        LaunchKernel<TypeMapType<Backend::kDeviceType, DataType::kFloat32>>(
            input, weight, bias, out, stream);
        break;
      default:
        throw std::invalid_argument("CUDA LayerNorm unsupported dtype");
    }
  }

 private:
  template <typename T>
  void LaunchKernel(const Tensor input, const Tensor weight,
                    const std::optional<Tensor> bias, Tensor out,
                    typename Backend::Stream stream) const {
    CudaLayerNormKernel<256, Backend::kDeviceType, T>
        <<<static_cast<unsigned>(rows_), 256, 0, stream>>>(
            static_cast<const T*>(input.data()),
            static_cast<const T*>(weight.data()),
            bias ? static_cast<const T*>(bias->data()) : nullptr,
            static_cast<T*>(out.data()), layout_, eps_);
  }

  static Tensor checked(const Tensor input, const Tensor weight,
                        const std::optional<Tensor> bias, float eps,
                        Tensor out) {
    if (!supports(input, weight, bias, eps, out)) {
      throw std::invalid_argument(
          "unsupported CUDA LayerNorm metadata, device or alias");
    }
    return input;
  }

  static std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  }

  CudaLayerNormLayout layout_{};
  std::size_t rows_ = 1;
  std::size_t input_bytes_ = 0;
  std::size_t weight_bytes_ = 0;
  std::size_t out_bytes_ = 0;
  std::size_t bias_bytes_ = 0;
  bool exact_inplace_layout_ = false;
};

}  // namespace infini::ops

#endif
