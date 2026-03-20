#ifndef INFINI_OPS_COMMON_CUDA_CASTER_H_
#define INFINI_OPS_COMMON_CUDA_CASTER_H_

#ifdef WITH_NVIDIA
#include <cuda_runtime.h>
#elif defined(WITH_ILUVATAR)
#include <cuda_runtime.h>
#elif defined(WITH_HYGON)
#include <cuda_runtime.h>
#elif defined(WITH_METAX)
#include <mcr/mc_runtime.h>
#elif defined(WITH_MOORE)
#include <musa_runtime.h>
#endif

#include "caster.h"

namespace infini::ops {

template <>
struct Caster<kDeviceType> {
  template <typename Dst, typename Src>
  __host__ __device__ static Dst Cast(Src&& x) {
    static_assert(!std::is_reference_v<Dst>,
                  "`Cast` cannot return reference types");

    using PureSrc = std::remove_cv_t<std::remove_reference_t<Src>>;
    using PureDst = std::remove_cv_t<std::remove_reference_t<Dst>>;

    if constexpr (std::is_same_v<PureSrc, PureDst>) {
      return std::forward<Src>(x);
    } else {
      return HardwareCast<PureDst>(std::forward<Src>(x), PriorityHigh{});
    }
  }

 private:
  template <typename T>
  using PureType = std::remove_cv_t<std::remove_reference_t<T>>;

  template <typename T>
  __host__ __device__ static constexpr float ToFloatHelper(T&& x) {
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
  __host__ __device__ static constexpr Dst FromFloatHelper(float f) {
    using PureDst = PureType<Dst>;
    if constexpr (IsBFloat16<PureDst>) {
      return __float2bfloat16(f);
    } else if constexpr (IsFP16<PureDst>) {
      return __float2half(f);
    } else {
      return static_cast<Dst>(f);
    }
  }

  // Priority tags for overload resolution.
  struct PriorityLow {};

  struct PriorityHigh : PriorityLow {};

  // Fallback: lowest priority. This always matches if nothing else does.
  template <typename Dst, typename Src>
  __host__ __device__ static constexpr Dst HardwareCast(Src&& x, PriorityLow) {
    return FromFloatHelper<Dst>(ToFloatHelper(std::forward<Src>(x)));
  }

// Usage: `DEFINE_DIRECT_CAST(INTRINSIC, CONDITION)`.
#define DEFINE_DIRECT_CAST(INTRINSIC, ...)                            \
  template <typename Dst, typename Src>                               \
  __host__ __device__ static auto HardwareCast(Src x, PriorityHigh)   \
      -> std::enable_if_t<(__VA_ARGS__),                              \
                          decltype(INTRINSIC(std::declval<Src>()))> { \
    return INTRINSIC(x);                                              \
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
};

template <typename Dst, typename... Args>
__host__ __device__ __forceinline__ auto Cast(Args&&... args) {
  return Caster<kDeviceType>::template Cast<Dst>(std::forward<Args>(args)...);
}

}  // namespace infini::ops

#endif
