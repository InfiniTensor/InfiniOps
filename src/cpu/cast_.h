#ifndef INFINI_OPS_COMMON_CPU_CAST_H_
#define INFINI_OPS_COMMON_CPU_CAST_H_

#include <type_traits>

#include "cast.h"

namespace infini::ops {

namespace detail {

template <typename T, typename = void>
struct HasToFloat : std::false_type {};

template <typename T>
struct HasToFloat<T, std::void_t<decltype(std::declval<T>().ToFloat())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasFromFloat : std::false_type {};

template <typename T>
struct HasFromFloat<T,
                    std::void_t<decltype(T::FromFloat(std::declval<float>()))>>
    : std::true_type {};

template <typename T>
constexpr float ToFloatHelper(T&& x) {
  if constexpr (HasToFloat<T>::value) {
    return std::forward<T>(x).ToFloat();
  } else {
    return static_cast<float>(x);
  }
}

template <typename PureDst>
constexpr PureDst FromFloatHelper(float f) {
  if constexpr (HasFromFloat<PureDst>::value) {
    return PureDst::FromFloat(f);
  } else {
    return static_cast<PureDst>(f);
  }
}

}  // namespace detail

template <>
struct Caster<Device::Type::kCpu> {
  template <typename Dst, typename Src>
  static Dst Cast(Src&& x) {
    static_assert(!std::is_reference_v<Dst>,
                  "`Cast` cannot return reference types");

    using PureDst = std::remove_cv_t<std::remove_reference_t<Dst>>;
    using PureSrc = std::remove_cv_t<std::remove_reference_t<Src>>;

    if constexpr (std::is_same_v<PureDst, PureSrc>) {
      return std::forward<Src>(x);
    }

    constexpr bool src_is_custom = IsBFloat16<PureSrc> || IsFP16<PureSrc>;
    constexpr bool dst_is_custom = IsBFloat16<PureDst> || IsFP16<PureDst>;

    if constexpr (!src_is_custom && !dst_is_custom) {
      return static_cast<PureDst>(std::forward<Src>(x));
    } else {
      return detail::FromFloatHelper<PureDst>(
          detail::ToFloatHelper(std::forward<Src>(x)));
    }
  }
};

}  // namespace infini::ops

#endif
