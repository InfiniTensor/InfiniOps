#ifndef INFINI_OPS_KUNLUN_BLAS_H_
#define INFINI_OPS_KUNLUN_BLAS_H_

#include <utility>

// clang-format off
// XBLAS exposes the cublas API via `#define cublas* xblas*` macros; including
// `cublas_api.h` brings `cublasHandle_t`, `cublasCreate`, and the GemmEx
// entry points, all resolved at link time to `libxpu_blas.so` symbols.
// `library_types.h` (XRE) provides the `cudaDataType_t` enum values.
#include "cublas_api.h"
#include "library_types.h"
// clang-format on

#include "data_type.h"
#include "native/cuda/blas.h"
#include "native/kunlun/runtime_.h"

namespace infini::ops {

template <>
struct Blas<Device::Type::kKunlun> : public Runtime<Device::Type::kKunlun> {
  using BlasHandle = cublasHandle_t;

  static constexpr auto BLAS_OP_N = CUBLAS_OP_N;

  static constexpr auto BLAS_OP_T = CUBLAS_OP_T;

  static constexpr auto R_16F = CUDA_R_16F;

  static constexpr auto R_16BF = CUDA_R_16BF;

  static constexpr auto R_32F = CUDA_R_32F;

  // XBLAS has no TF32; plain 32F compute.
  static constexpr auto BLAS_COMPUTE_32F = CUBLAS_COMPUTE_32F;

  static constexpr auto BLAS_COMPUTE_32F_FAST_TF32 = CUBLAS_COMPUTE_32F;

  static constexpr auto BLAS_GEMM_DEFAULT = CUBLAS_GEMM_DEFAULT;

  static constexpr auto BlasCreate = cublasCreate_v2;

  static constexpr auto BlasSetStream = cublasSetStream_v2;

  static constexpr auto BlasDestroy = cublasDestroy_v2;

  static constexpr auto BlasGemmStridedBatchedEx = [](auto&&... args) {
    return cublasGemmStridedBatchedEx(std::forward<decltype(args)>(args)...);
  };

  // xBLAS's strided-batched GEMM path rejects bf16 inputs entirely (status
  // 15, NOT_SUPPORTED) while single cublasGemmEx accepts AB=bf16 with C=f32.
  // The bf16 wrapper loops over the batch with GemmEx instead.
  static constexpr auto BlasGemmEx = [](auto&&... args) {
    return cublasGemmEx(std::forward<decltype(args)>(args)...);
  };
};

}  // namespace infini::ops

#endif
