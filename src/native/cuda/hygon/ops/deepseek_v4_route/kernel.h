#pragma once
#include <cuda_runtime.h>

#include "base/deepseek_v4_route.h"
#include "kernel.cuh"

namespace infini::ops {

template <>
class Operator<DeepseekV4Route, Device::Type::kHygon> : public DeepseekV4Route {
 public:
  using Opt = std::optional<Tensor>;
  using DeepseekV4Route::DeepseekV4Route;
  bool capture_safe() const override { return true; }
  static bool supports(Tensor x, Opt b, Opt t, Opt h, double s, Tensor w,
                       Tensor i) {
    if (x.ndim() != 2 || !x.size(0) || !x.size(1) || w.ndim() != 2 ||
        w.size(0) != x.size(0) || !w.size(1) || w.size(1) > x.size(1) ||
        w.shape() != i.shape() || x.dtype() != DataType::kFloat32 ||
        w.dtype() != DataType::kFloat32 || i.dtype() != DataType::kInt64 ||
        x.device().type() != Device::Type::kHygon || !std::isfinite(s) ||
        s <= 0 || bool(t) != bool(h) || (t && b))
      return false;
    if (b && (b->shape() != Tensor::Shape{x.size(1)} ||
              b->dtype() != DataType::kFloat32))
      return false;
    if (t &&
        (t->shape() != Tensor::Shape{x.size(0)} ||
         (t->dtype() != DataType::kInt32 && t->dtype() != DataType::kInt64) ||
         h->ndim() != 2 || !h->size(0) || h->size(1) != w.size(1) ||
         h->dtype() != DataType::kInt64))
      return false;
    for (const auto& a : {x, w, i})
      if (a.device() != x.device() || !a.IsContiguous()) return false;
    for (const auto& a : {b, t, h})
      if (a && (a->device() != x.device() || !a->IsContiguous())) return false;
    return true;
  }
  void operator()(Tensor x, Opt b, Opt t, Opt h, double s, Tensor w,
                  Tensor i) const override {
    if (!supports(x, b, t, h, s, w, i))
      throw std::invalid_argument("Invalid DeepSeek V4 routing inputs");
    auto launch = [&](auto tag) {
      using T = decltype(tag);
      v4_route::route<T><<<static_cast<unsigned>(std::min<size_t>(
                               (x.size(0) - 1) / 64 + 1, 65535)),
                           64, 0, static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const float*>(x.data()),
          b ? static_cast<const float*>(b->data()) : nullptr,
          t ? static_cast<const T*>(t->data()) : nullptr,
          h ? static_cast<const int64_t*>(h->data()) : nullptr, float(s),
          static_cast<float*>(w.data()), static_cast<int64_t*>(i.data()),
          x.size(0), x.size(1), w.size(1), h ? h->size(0) : 0);
    };
    if (!h && x.size(1) == 256 && w.size(1) == 6) {
      v4_route::RouteWave64<<<static_cast<unsigned>(std::min<size_t>(
                                  (x.size(0) - 1) / 4 + 1, 65535)),
                              256, 0, static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const float*>(x.data()),
          b ? static_cast<const float*>(b->data()) : nullptr, float(s),
          static_cast<float*>(w.data()), static_cast<int64_t*>(i.data()),
          x.size(0));
    } else if (t && t->dtype() == DataType::kInt32)
      launch(int32_t{});
    else
      launch(int64_t{});
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("DeepSeek V4 routing launch failed");
  }
};
}  // namespace infini::ops
