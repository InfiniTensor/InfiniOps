#include "native/cuda/hygon/dynamic_int8_quant.cuh"
#include "native/cuda/hygon/ops/deepseek_v4_route/kernel.cuh"
#include "native/cuda/hygon/ops/i8_gemm/gemv_wave64.cuh"
#include "native/cuda/hygon/ops/per_channel_quant_i8/quantize_fixed.cuh"

extern "C" int check_route(const float* x, const float* bias, float* w,
                           int64_t* ids, size_t rows, float scale,
                           int optimized, void* stream) {
  using namespace infini::ops::v4_route;
  const auto s = static_cast<cudaStream_t>(stream);
  if (optimized) {
    RouteWave64<<<(rows + 3) / 4, 256, 0, s>>>(x, bias, scale, w, ids, rows);
  } else {
    route<int64_t><<<(rows + 63) / 64, 64, 0, s>>>(
        x, bias, nullptr, nullptr, scale, w, ids, rows, 256, 6, 0);
  }
  return static_cast<int>(cudaGetLastError());
}

template <class T>
void quant(const void* x, int8_t* packed, float* scales, size_t rows, int width,
           int optimized, cudaStream_t stream) {
  using namespace infini::ops::hygon_dynamic_int8;
  const unsigned blocks = rows > 65535 ? 65535 : rows;
  if (optimized && width == 1024) {
    QuantizeFixed<T, 1024><<<blocks, 256, 0, stream>>>(static_cast<const T*>(x),
                                                       packed, scales, rows);
  } else if (optimized && width == 4096) {
    QuantizeFixed<T, 4096><<<blocks, 256, 0, stream>>>(static_cast<const T*>(x),
                                                       packed, scales, rows);
  } else {
    Quantize<T, true><<<blocks, kThreads, 0, stream>>>(
        static_cast<const T*>(x), packed, scales, rows, width);
  }
}

extern "C" int check_quant(const void* x, int8_t* packed, float* scales,
                           size_t rows, int width, int bf16, int optimized,
                           void* stream) {
  if (bf16)
    quant<__nv_bfloat16>(x, packed, scales, rows, width, optimized,
                         static_cast<cudaStream_t>(stream));
  else
    quant<__half>(x, packed, scales, rows, width, optimized,
                  static_cast<cudaStream_t>(stream));
  return static_cast<int>(cudaGetLastError());
}

template <class T>
void gemv(void* out, const void* bias, const int8_t* x, const float* xs,
          const int8_t* w, const float* ws, int n, int width, int vectorized,
          ptrdiff_t bias_col, cudaStream_t stream) {
  using namespace infini::ops::hygon_i8_gemm;
  if (width == 1024) {
    gemv_i8_hygon<1024, T><<<(n + 7) / 8, 512, 0, stream>>>(
        static_cast<T*>(out), static_cast<const T*>(bias), x, xs, w, ws, n,
        bias_col);
  } else if (vectorized) {
    gemv_vector_hygon<4096, T><<<(n + 7) / 8, 512, 0, stream>>>(
        static_cast<T*>(out), static_cast<const T*>(bias), x, xs, w, ws, n,
        bias_col);
  } else {
    gemv_i8_hygon<4096, T><<<(n + 3) / 4, 256, 0, stream>>>(
        static_cast<T*>(out), static_cast<const T*>(bias), x, xs, w, ws, n,
        bias_col);
  }
}

extern "C" int check_gemv(void* out, const void* bias, const int8_t* x,
                          const float* xs, const int8_t* w, const float* ws,
                          int n, int width, int dtype, int vectorized,
                          ptrdiff_t bias_col, void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  if (dtype == 0)
    gemv<float>(out, bias, x, xs, w, ws, n, width, vectorized, bias_col, s);
  else if (dtype == 1)
    gemv<__half>(out, bias, x, xs, w, ws, n, width, vectorized, bias_col, s);
  else
    gemv<__nv_bfloat16>(out, bias, x, xs, w, ws, n, width, vectorized, bias_col,
                        s);
  return static_cast<int>(cudaGetLastError());
}
