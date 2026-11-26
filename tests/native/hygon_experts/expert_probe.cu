#include <infini/ops.h>

#include "native/cuda/hygon/ops/deepseek_v4_w8a8_experts/kernel.h"
template <class T>
int Run(const void* x, const float* weights, const int64_t* ids,
        const int8_t* gate, const float* gate_scales, const int8_t* down,
        const float* down_scales, void* output, void* workspace, size_t m,
        size_t h, size_t inner, size_t experts, size_t topk, float limit,
        int routed, float routed_scale, int optimized, cudaStream_t stream) {
  using namespace infini::ops;
  using namespace deepseek_v4_w8a8_detail;
  const size_t routes = m * topk;
  auto* activated = static_cast<float*>(workspace);
  auto* xs = activated + routes * inner;
  auto* acts = xs + m;
  auto* qx = reinterpret_cast<int8_t*>(acts + routes);
  auto* qa = qx + m * h;
  const auto grid = [](size_t n) {
    return static_cast<unsigned>(n < 65535 ? n : 65535);
  };
  if (optimized) {
    const Device dev{Device::Type::kHygon, 0};
    const DataType dtype =
        std::is_same_v<T, float> ? DataType::kFloat32 : DataType::kBFloat16;
    Tensor tx(const_cast<void*>(x), {m, h}, dtype, dev),
        tw(const_cast<float*>(weights), {m, topk}, DataType::kFloat32, dev);
    Tensor ti(const_cast<int64_t*>(ids), {m, topk}, DataType::kInt64, dev);
    Tensor tg(const_cast<int8_t*>(gate), {experts, 2 * inner, h},
              DataType::kInt8, dev);
    Tensor tgs(const_cast<float*>(gate_scales), {experts, 2 * inner, 1},
               DataType::kFloat32, dev);
    Tensor td(const_cast<int8_t*>(down), {experts, h, inner}, DataType::kInt8,
              dev);
    Tensor tds(const_cast<float*>(down_scales), {experts, h, 1},
               DataType::kFloat32, dev);
    Tensor ty(output, {m, h}, dtype, dev);
    Handle handle;
    handle.set_stream(stream);
    handle.set_workspace(workspace);
    handle.set_workspace_size_in_bytes(
        (routes * inner + m + routes) * sizeof(float) + m * h + routes * inner);
    const auto rs = routed ? std::optional<double>(routed_scale) : std::nullopt;
    DeepseekV4W8a8Experts::Call(
        handle, DeepseekV4W8a8Experts::DefaultConfig(dev.type()), tx, tw, ti,
        tg, tgs, td, tds, double(limit), rs, ty);
  } else {
    Quantize<T, true><<<grid(m), kThreads, 0, stream>>>(
        static_cast<const T*>(x), qx, xs, m, h);
    GateUp<T><<<grid(routes * inner), kThreads, 0, stream>>>(
        qx, xs, ids, gate, gate_scales, activated, routes, topk, experts, h,
        inner, limit, routed);
    Quantize<float, true><<<grid(routes), kThreads, 0, stream>>>(
        activated, qa, acts, routes, inner);
    deepseek_v4_w8a8_detail::Down<T><<<grid(m * h), kThreads, 0, stream>>>(
        qa, acts, ids, weights, down, down_scales, static_cast<T*>(output), m,
        topk, experts, h, inner, routed, routed_scale);
  }
  return static_cast<int>(cudaGetLastError());
}
extern "C" int run(const void* x, const float* weights, const int64_t* ids,
                   const int8_t* gate, const float* gate_scales,
                   const int8_t* down, const float* down_scales, void* output,
                   void* workspace, size_t m, size_t h, size_t inner,
                   size_t experts, size_t topk, float limit, int routed,
                   float routed_scale, int optimized, int bf16, void* stream) {
  try {
    if (bf16)
      return Run<__nv_bfloat16>(x, weights, ids, gate, gate_scales, down,
                                down_scales, output, workspace, m, h, inner,
                                experts, topk, limit, routed, routed_scale,
                                optimized, static_cast<cudaStream_t>(stream));
    return Run<float>(x, weights, ids, gate, gate_scales, down, down_scales,
                      output, workspace, m, h, inner, experts, topk, limit,
                      routed, routed_scale, optimized,
                      static_cast<cudaStream_t>(stream));
  } catch (const std::invalid_argument&) {
    return 101;
  } catch (...) {
    return 102;
  }
}
