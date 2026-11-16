#pragma once
#include "base/per_channel_quant_i8.h"
#include "native/cuda/hygon/dynamic_int8_quant.cuh"

namespace infini::ops {

template <>
class Operator<PerChannelQuantI8, Device::Type::kHygon> : public PerChannelQuantI8 {
 public:
  using PerChannelQuantI8::PerChannelQuantI8;
  bool capture_safe() const override { return true; }
  static bool supports(Tensor x, Tensor packed, Tensor scale) {
    return x.device().type() == Device::Type::kHygon && packed.device() == x.device() &&
           scale.device() == x.device() && x.ndim() == 2 && x.size(1) > 0 &&
           (x.dtype() == DataType::kBFloat16 || x.dtype() == DataType::kFloat32 ||
            x.dtype() == DataType::kFloat16) && packed.dtype() == DataType::kInt8 &&
           scale.dtype() == DataType::kFloat32 && packed.shape() == x.shape() &&
           scale.shape() == Tensor::Shape{x.size(0), 1} &&
           x.IsContiguous() && packed.IsContiguous() && scale.IsContiguous();
  }
  void operator()(Tensor x, Tensor packed, Tensor scale) const override {
    if (!supports(x, packed, scale)) throw std::invalid_argument("Unsupported Hygon INT8 quantization");
    if (!x.size(0)) return;
    using namespace hygon_dynamic_int8;
    auto launch = [&](auto tag) {
      using T = decltype(tag);
      Quantize<T, true><<<static_cast<unsigned>(std::min<size_t>(x.size(0), 65535)),
                    kThreads, 0, static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const T*>(x.data()), static_cast<int8_t*>(packed.data()),
          static_cast<float*>(scale.data()), x.size(0), x.size(1));
    };
    if (x.dtype() == DataType::kFloat32) launch(float{});
    else if (x.dtype() == DataType::kFloat16) launch(__half{});
    else launch(__nv_bfloat16{});
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("Hygon INT8 quantization launch failed");
  }
};

} // namespace infini::ops
