#ifndef INFINI_OPS_COMMON_CAST_H_
#define INFINI_OPS_COMMON_CAST_H_

#include "data_type.h"

namespace infini::ops {

namespace detail {

template <typename T>
constexpr float ToFloatHelper(T &&x) {
  using PureSrc = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (IsBFloat16<PureSrc> || IsFP16<PureSrc>) {
    return std::forward<T>(x).ToFloat();
  } else {
    return static_cast<float>(std::forward<T>(x));
  }
}

template <typename Dst>
constexpr Dst FromFloatHelper(float f) {
  using PureDst = std::remove_cv_t<std::remove_reference_t<Dst>>;
  if constexpr (IsBFloat16<PureDst> || IsFP16<PureDst>) {
    return PureDst::FromFloat(f);
  } else {
    return static_cast<Dst>(f);
  }
}

}  // namespace detail

template <typename Dst, typename Src>
Dst Cast(Src &&x) {
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

}  // namespace infini::ops

#endif
