#pragma once

#include "base/per_tensor_quant_i8.h"
#include "native/cuda/hygon/per_tensor_i8.cuh"

namespace infini::ops {

template <>
class Operator<PerTensorQuantI8, Device::Type::kHygon>
    : public PerTensorQuantI8 {
 public:
  using PerTensorQuantI8::PerTensorQuantI8;
  bool capture_safe() const override { return true; }

  static bool supports(Tensor x, Tensor packed, Tensor scale,
                       std::optional<Tensor> zero, bool is_static) {
    using namespace hygon_tensor_i8;
    return !zero && x.device().type() == Device::Type::kHygon && IsFloat(x) &&
           packed.dtype() == DataType::kInt8 && packed.device() == x.device() &&
           packed.shape() == x.shape() && ScalarOnDevice(scale, x) &&
           ValidLayout(x, false) && ValidLayout(packed, true);
  }

  void operator()(Tensor x, Tensor packed, Tensor scale,
                  std::optional<Tensor> zero, bool is_static) const override {
    using namespace hygon_tensor_i8;
    if (!supports(x, packed, scale, zero, is_static))
      throw std::invalid_argument(
          "Unsupported Hygon per-tensor INT8 quantization");
    const auto stream = static_cast<cudaStream_t>(stream_);
    const auto count = x.numel();
    const auto in_layout = MakeLayout(x), out_layout = MakeLayout(packed);
    auto launch = [&](auto tag) {
      using T = decltype(tag);
      const auto* input = static_cast<const T*>(x.data());
      auto* output = static_cast<int8_t*>(packed.data());
      auto* scales = static_cast<float*>(scale.data());
      if (!is_static && count <= 4096) {
        DynamicSmall<T><<<1, kThreads, 0, stream>>>(
            input, output, scales, count, in_layout, out_layout);
        return;
      }
      const unsigned blocks = count ? static_cast<unsigned>(std::min<size_t>(
                                          (count - 1) / kThreads + 1, 65535))
                                    : 0;
      if (!is_static) {
        if (cudaMemsetAsync(scales, 0, sizeof(float), stream) != cudaSuccess)
          throw std::runtime_error("Hygon quantization scale reset failed");
        AbsMaximum<T>
            <<<blocks, kThreads, 0, stream>>>(input, scales, count, in_layout);
        FinalizeScale<<<1, 1, 0, stream>>>(scales);
      }
      if (count)
        Quantize<T><<<blocks, kThreads, 0, stream>>>(
            input, output, scales, count, in_layout, out_layout);
    };
    if (x.dtype() == DataType::kFloat32)
      launch(float{});
    else if (x.dtype() == DataType::kFloat16)
      launch(__half{});
    else
      launch(__nv_bfloat16{});
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("Hygon per-tensor quantization launch failed");
  }
};

}  // namespace infini::ops
