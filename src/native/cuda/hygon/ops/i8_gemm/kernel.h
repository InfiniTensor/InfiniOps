#pragma once
#include <cuda_runtime.h>

#include <type_traits>

#include "base/i8_gemm.h"
#include "gemv_wave64.cuh"
#include "native/cuda/hygon/caster.cuh"

namespace infini::ops {
namespace hygon_i8_gemm {

template <class T>
__global__ void matmul(T* out, const int8_t* a, const float* as,
                       const int8_t* b, const float* bs, const T* bias,
                       size_t m, size_t n, size_t k, ptrdiff_t bias_row,
                       ptrdiff_t bias_col) {
  __shared__ int32_t partial[128];
  for (size_t idx = blockIdx.x; idx < m * n; idx += gridDim.x) {
    const size_t row = idx / n, col = idx % n;
    int32_t sum = 0;
    for (size_t j = threadIdx.x; j < k; j += blockDim.x)
      sum += int32_t(a[row * k + j]) * b[col * k + j];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned step = 64; step; step /= 2) {
      if (threadIdx.x < step)
        partial[threadIdx.x] += partial[threadIdx.x + step];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      using Convert = Caster<Device::Type::kHygon>;
      // The W8A8 epilogue rounds the joint scale before applying it to the dot.
      const float scale = __fmul_rn(as[row], bs[col]);
      float value = __fmul_rn(static_cast<float>(partial[0]), scale);
      if (bias)
        value += Convert::template Cast<float>(
            bias[row * bias_row + col * bias_col]);
      out[idx] = Convert::template Cast<T>(value);
    }
    __syncthreads();
  }
}

}  // namespace hygon_i8_gemm

template <>
class Operator<I8Gemm, Device::Type::kHygon> : public I8Gemm {
 public:
  using I8Gemm::I8Gemm;
  bool capture_safe() const override { return true; }
  static bool supports(Tensor c, Tensor a, Tensor as, Tensor b, Tensor bs,
                       const std::optional<Tensor> bias) {
    if (a.ndim() != 2 || b.ndim() != 2 || c.ndim() != 2 || a.size(1) == 0 ||
        a.size(1) > INT32_MAX / (128 * 128) || b.size(0) != a.size(1) ||
        a.dtype() != DataType::kInt8 || b.dtype() != DataType::kInt8 ||
        as.dtype() != DataType::kFloat32 || bs.dtype() != DataType::kFloat32 ||
        (c.dtype() != DataType::kFloat32 && c.dtype() != DataType::kBFloat16 &&
         c.dtype() != DataType::kFloat16) ||
        c.shape() != Tensor::Shape{a.size(0), b.size(1)} ||
        as.shape() != Tensor::Shape{a.size(0), 1} ||
        bs.shape() != Tensor::Shape{b.size(1), 1} || !a.IsContiguous() ||
        !c.IsContiguous() || !as.IsContiguous() || !bs.IsContiguous() ||
        b.stride(0) != 1 || b.stride(1) != static_cast<ptrdiff_t>(b.size(0)) ||
        c.device().type() != Device::Type::kHygon || a.device() != c.device() ||
        b.device() != c.device() || as.device() != c.device() ||
        bs.device() != c.device())
      return false;
    return !bias || (bias->device() == c.device() &&
                     bias->dtype() == c.dtype() && bias->shape() == c.shape() &&
                     bias->stride(0) >= 0 && bias->stride(1) >= 0);
  }
  void operator()(Tensor c, Tensor a, Tensor as, Tensor b, Tensor bs,
                  const std::optional<Tensor> bias) const override {
    if (!supports(c, a, as, b, bs, bias))
      throw std::invalid_argument("Unsupported Hygon INT8 GEMM");
    if (!c.numel()) return;
    auto launch = [&](auto tag) {
      using T = decltype(tag);
      if (a.size(0) == 1 && (a.size(1) == 1024 || a.size(1) == 4096) &&
          c.size(1) <= INT32_MAX - 8 &&
          (reinterpret_cast<uintptr_t>(a.data()) & 3u) == 0 &&
          (reinterpret_cast<uintptr_t>(b.data()) & 3u) == 0) {
        auto run = [&](auto width, auto vectorized) {
          constexpr int k = decltype(width)::value;
          const int n = static_cast<int>(c.size(1));
          const int threads =
              ((k == 1024 && n == 4096) || vectorized) ? 512 : 256;
          const int blocks = (n + threads / 64 - 1) / (threads / 64);
          if constexpr (decltype(vectorized)::value) {
            hygon_i8_gemm::gemv_vector_hygon<k, T>
                <<<blocks, threads, 0, static_cast<cudaStream_t>(stream_)>>>(
                    static_cast<T*>(c.data()),
                    bias ? static_cast<const T*>(bias->data()) : nullptr,
                    static_cast<const int8_t*>(a.data()),
                    static_cast<const float*>(as.data()),
                    static_cast<const int8_t*>(b.data()),
                    static_cast<const float*>(bs.data()), n,
                    bias ? bias->stride(1) : 0);
          } else {
            hygon_i8_gemm::gemv_i8_hygon<k, T>
                <<<blocks, threads, 0, static_cast<cudaStream_t>(stream_)>>>(
                    static_cast<T*>(c.data()),
                    bias ? static_cast<const T*>(bias->data()) : nullptr,
                    static_cast<const int8_t*>(a.data()),
                    static_cast<const float*>(as.data()),
                    static_cast<const int8_t*>(b.data()),
                    static_cast<const float*>(bs.data()), n,
                    bias ? bias->stride(1) : 0);
          }
        };
        if (a.size(1) == 1024) {
          run(std::integral_constant<int, 1024>{}, std::false_type{});
        } else if (c.size(1) == 1280 &&
                   (reinterpret_cast<uintptr_t>(a.data()) & 15u) == 0 &&
                   (reinterpret_cast<uintptr_t>(b.data()) & 15u) == 0) {
          run(std::integral_constant<int, 4096>{}, std::true_type{});
        } else {
          run(std::integral_constant<int, 4096>{}, std::false_type{});
        }
        return;
      }
      hygon_i8_gemm::matmul<T>
          <<<static_cast<unsigned>(std::min<size_t>(c.numel(), 65535)), 128, 0,
             static_cast<cudaStream_t>(stream_)>>>(
              static_cast<T*>(c.data()), static_cast<const int8_t*>(a.data()),
              static_cast<const float*>(as.data()),
              static_cast<const int8_t*>(b.data()),
              static_cast<const float*>(bs.data()),
              bias ? static_cast<const T*>(bias->data()) : nullptr, c.size(0),
              c.size(1), a.size(1), bias ? bias->stride(0) : 0,
              bias ? bias->stride(1) : 0);
    };
    if (c.dtype() == DataType::kFloat32)
      launch(float{});
    else if (c.dtype() == DataType::kFloat16)
      launch(__half{});
    else
      launch(__nv_bfloat16{});
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("Hygon INT8 GEMM launch failed");
  }
};

}  // namespace infini::ops
