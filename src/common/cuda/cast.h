#ifndef INFINI_OPS_COMMON_CUDA_CAST_H_
#define INFINI_OPS_COMMON_CUDA_CAST_H_

#ifdef WITH_NVIDIA
#include <cuda_runtime.h>
#elif WITH_METAX
#include <mcr/mc_runtime.h>
#endif

#include "data_type.h"

namespace infini::ops {

namespace detail {

template <typename T>
using PureType = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
__host__ __device__ constexpr float ToFloatHelper(T&& x) {
  using PureSrc = PureType<T>;
  if constexpr (IsBFloat16<PureSrc>) {
    return __bfloat162float(x);
  } else if constexpr (IsFP16<PureSrc>) {
    return __half2float(x);
  } else {
    return static_cast<float>(std::forward<T>(x));
  }
}

template <typename Dst>
__host__ __device__ constexpr Dst FromFloatHelper(float f) {
  using PureDst = PureType<Dst>;
  if constexpr (IsBFloat16<PureDst>) {
    return __float2bfloat16(f);
  } else if constexpr (IsFP16<PureDst>) {
    return __float2half(f);
  } else {
    return static_cast<Dst>(f);
  }
}

// Priority tags for overload resolution
struct PriorityLow {};
struct PriorityHigh : PriorityLow {};

// Fallback: Lowest priority, always matches if nothing else does.
template <typename Dst, typename Src>
__host__ __device__ constexpr Dst HardwareCast(Src&& x, PriorityLow) {
  return FromFloatHelper<Dst>(ToFloatHelper(std::forward<Src>(x)));
}

// Usage: DEFINE_DIRECT_CAST(INTRINSIC, CONDITION)
#define DEFINE_DIRECT_CAST(INTRINSIC, ...)                           \
  template <typename Dst, typename Src>                              \
  __host__ __device__ auto HardwareCast(Src x, PriorityHigh)         \
      ->std::enable_if_t<(__VA_ARGS__),                              \
                         decltype(INTRINSIC(std::declval<Src>()))> { \
    return INTRINSIC(x);                                             \
  }

DEFINE_DIRECT_CAST(
    __bfloat162int_rn,
    std::is_same_v<PureType<Dst>, int>&& IsBFloat16<PureType<Src>>)
DEFINE_DIRECT_CAST(
    __bfloat162short_rn,
    std::is_same_v<PureType<Dst>, short>&& IsBFloat16<PureType<Src>>)
DEFINE_DIRECT_CAST(
    __int2bfloat16_rn,
    IsBFloat16<PureType<Dst>>&& std::is_same_v<PureType<Src>, int>)
DEFINE_DIRECT_CAST(__int2half_rn,
                   IsFP16<PureType<Dst>>&& std::is_same_v<PureType<Src>, int>)
DEFINE_DIRECT_CAST(
    __double2bfloat16,
    IsBFloat16<PureType<Dst>>&& std::is_same_v<PureType<Src>, double>)
DEFINE_DIRECT_CAST(
    __double2half,
    IsFP16<PureType<Dst>>&& std::is_same_v<PureType<Src>, double>)
DEFINE_DIRECT_CAST(__half, IsFP16<PureType<Dst>>&& IsBFloat16<PureType<Src>>)
#undef DEFINE_DIRECT_CAST

}  // namespace detail

template <typename Dst, typename Src>
__host__ __device__ Dst Cast(Src&& x) {
  static_assert(!std::is_reference_v<Dst>,
                "`Cast` cannot return reference types");

  using PureSrc = std::remove_cv_t<std::remove_reference_t<Src>>;
  using PureDst = std::remove_cv_t<std::remove_reference_t<Dst>>;

  if constexpr (std::is_same_v<PureSrc, PureDst>) {
    return std::forward<Src>(x);
  } else {
    return detail::HardwareCast<PureDst>(std::forward<Src>(x),
                                         detail::PriorityHigh{});
  }
}

}  // namespace infini::ops

#endif
