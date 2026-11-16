#ifndef INFINI_OPS_HYGON_DEEPSEEK_V4_STORE_MLA_H_
#define INFINI_OPS_HYGON_DEEPSEEK_V4_STORE_MLA_H_

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <stdexcept>

#include "base/deepseek_v4_store_mla.h"
#include "native/cuda/hygon/caster.cuh"

namespace infini::ops {
namespace v4_store_mla_detail {

template <class T>
static __global__ void Store(const T* kv, const int32_t* slots, uint8_t* pages,
                      size_t rows, size_t page_count, size_t page_stride, size_t page_tokens) {
  using Convert = Caster<Device::Type::kHygon>;
  constexpr size_t nope_dim = 448, value_bytes = 576;
  __shared__ float max_abs[7][64];
  __shared__ float inverse_scale[7];
  __shared__ uint8_t exponents[7];
  const unsigned lane = threadIdx.x;
  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int32_t id = slots[row];
    if (id < 0 || static_cast<size_t>(id) / page_tokens >= page_count) continue;
    const auto* input = kv + row * 512;
    for (size_t i = lane; i < nope_dim; i += blockDim.x) {
      max_abs[i / 64][i % 64] = fabsf(Convert::template Cast<float>(input[i]));
    }
    __syncthreads();
    for (unsigned step = 32; step; step /= 2) {
      for (size_t i = lane; i < nope_dim; i += blockDim.x) {
        const auto channel = i % 64;
        if (channel < step) {
          auto* group = max_abs[i / 64];
          group[channel] = fmaxf(group[channel], group[channel + step]);
        }
      }
      __syncthreads();
    }
    if (lane < 7) {
      // Match the HCU FlashMLA writer's conservative E4M3 scale range.
      const float raw_scale = fmaxf(max_abs[lane][0], 1e-4f) / 224.0f;
      const uint32_t bits = __float_as_uint(raw_scale);
      const int encoded = min(int((bits >> 23) & 255) + int((bits & 0x7fffff) != 0), 254);
      exponents[lane] = static_cast<uint8_t>(encoded);
      inverse_scale[lane] = exp2f(127.0f - encoded);
    }
    __syncthreads();
    const auto offset = static_cast<size_t>(id) % page_tokens;
    auto* page = pages + static_cast<size_t>(id) / page_tokens * page_stride;
    auto* values = page + offset * value_bytes;
    for (size_t i = lane; i < nope_dim; i += blockDim.x) {
      values[i] = __nv_cvt_float_to_fp8(Convert::template Cast<float>(input[i]) * inverse_scale[i / 64],
                                         __NV_SATFINITE, __NV_E4M3);
    }
    if (lane < 64) {
      reinterpret_cast<__nv_bfloat16*>(values + nope_dim)[lane] =
          Convert::template Cast<__nv_bfloat16>(input[nope_dim + lane]);
    }
    if (lane < 8) page[page_tokens * value_bytes + offset * 8 + lane] = lane < 7 ? exponents[lane] : 0;
    __syncthreads();
  }
}
}  // namespace v4_store_mla_detail

template <>
class Operator<DeepseekV4StoreMla, Device::Type::kHygon> : public DeepseekV4StoreMla {
 public:
  static bool supports(const Tensor kv, const Tensor slots, int64_t page_tokens, const Tensor pages) {
    return kv.device().type() == Device::Type::kHygon &&
           kv.device() == slots.device() && kv.device() == pages.device() &&
           (kv.dtype() == DataType::kBFloat16 || kv.dtype() == DataType::kFloat32) &&
           slots.dtype() == DataType::kInt32 &&
           pages.dtype() == DataType::kUInt8 && kv.ndim() == 2 && kv.size(0) > 0 &&
           kv.size(1) == 512 && slots.shape() == Tensor::Shape{kv.size(0)} &&
           (page_tokens == 256 || page_tokens == 64 || page_tokens == 2) &&
           pages.ndim() == 2 && pages.size(0) > 0 &&
           pages.size(1) >= static_cast<size_t>(page_tokens) * 584 && pages.size(1) % 2 == 0 &&
           kv.IsContiguous() && slots.IsContiguous() && pages.IsContiguous();
  }

  Operator(const Tensor kv, const Tensor slots, int64_t page_tokens, Tensor pages)
      : DeepseekV4StoreMla(kv, slots, page_tokens, pages) {
    if (!supports(kv, slots, page_tokens, pages))
      throw std::invalid_argument("DeepSeek V4 raw MLA cache store inputs are unsupported");
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor kv, const Tensor slots, int64_t page_tokens, Tensor pages) const override {
    if (!supports(kv, slots, page_tokens, pages))
      throw std::invalid_argument("DeepSeek V4 raw MLA cache store inputs changed");
    const auto rows = kv.size(0);
    auto launch = [&](auto tag) {
      using T = decltype(tag);
      v4_store_mla_detail::Store<T><<<static_cast<unsigned>(rows < 65535 ? rows : 65535), 128, 0,
                                     static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const T*>(kv.data()), static_cast<const int32_t*>(slots.data()),
          static_cast<uint8_t*>(pages.data()), rows, pages.size(0), pages.size(1), page_tokens);
    };
    if (kv.dtype() == DataType::kFloat32) launch(float{});
    else launch(__nv_bfloat16{});
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("DeepSeek V4 cache store launch failed");
  }
};

}  // namespace infini::ops

#endif
