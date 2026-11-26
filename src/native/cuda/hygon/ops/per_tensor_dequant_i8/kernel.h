#pragma once

#include "base/per_tensor_dequant_i8.h"
#include "native/cuda/hygon/per_tensor_i8.cuh"

namespace infini::ops {

template <>
class Operator<PerTensorDequantI8, Device::Type::kHygon>
    : public PerTensorDequantI8 {
 public:
  using PerTensorDequantI8::PerTensorDequantI8;
  bool capture_safe() const override { return true; }

  static bool supports(Tensor x, Tensor packed, Tensor scale,
                       std::optional<Tensor> zero) {
    using namespace hygon_tensor_i8;
    return x.device().type() == Device::Type::kHygon && IsFloat(x) &&
           packed.dtype() == DataType::kInt8 && packed.device() == x.device() &&
           packed.shape() == x.shape() && ScalarOnDevice(scale, x) &&
           (!zero || ScalarOnDevice(*zero, x)) && ValidLayout(x, true) &&
           ValidLayout(packed, false);
  }

  void operator()(Tensor x, Tensor packed, Tensor scale,
                  std::optional<Tensor> zero) const override {
    using namespace hygon_tensor_i8;
    if (!supports(x, packed, scale, zero))
      throw std::invalid_argument(
          "Unsupported Hygon per-tensor INT8 dequantization");
    const auto count = x.numel();
    if (!count) return;
    const auto stream = static_cast<cudaStream_t>(stream_);
    const unsigned blocks = static_cast<unsigned>(
        std::min<size_t>((count - 1) / kThreads + 1, 65535));
    const auto in_layout = MakeLayout(packed), out_layout = MakeLayout(x);
    auto launch = [&](auto tag) {
      using T = decltype(tag);
      Dequantize<T><<<blocks, kThreads, 0, stream>>>(
          static_cast<T*>(x.data()), static_cast<const int8_t*>(packed.data()),
          static_cast<const float*>(scale.data()),
          zero ? static_cast<const float*>(zero->data()) : nullptr, count,
          out_layout, in_layout);
    };
    if (x.dtype() == DataType::kFloat32)
      launch(float{});
    else if (x.dtype() == DataType::kFloat16)
      launch(__half{});
    else
      launch(__nv_bfloat16{});
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("Hygon per-tensor dequantization launch failed");
  }
};

}  // namespace infini::ops
