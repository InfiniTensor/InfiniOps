#ifndef INFINI_OPS_KUNLUN_CASTER__H_
#define INFINI_OPS_KUNLUN_CASTER__H_

#include <infini/rt/kunlun/data_type_.h>

#include <type_traits>

#include "caster.h"
#include "data_type.h"

namespace infini::ops {

// Kunlun XPU3 has native fp16 (`half` from the XTDK) but no bf16 type: bf16
// payloads are `uint16_t` in memory and are computed in float with explicit
// bit manipulation on device.
namespace detail {

template <typename T>
struct IsKunlunFp16 : std::is_same<T, ::infini::rt::kunlun_float16> {};

inline float Fp16BitsToFloat(std::uint16_t h) {
  const std::uint32_t sign = (h & 0x8000u) << 16;
  const std::uint32_t exp = (h & 0x7C00u) >> 10;
  const std::uint32_t man = (h & 0x03FFu);
  std::uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      // Subnormal: normalize.
      std::uint32_t e = 127 - 15 + 1;
      while ((man & 0x0400u) == 0) {
        man <<= 1;
        e--;
      }
      man &= 0x03FFu;
      bits = sign | ((e) << 23) | (man << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline std::uint16_t FloatToFp16Bits(float f) {
  std::uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::int32_t exp =
      static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
  const std::uint32_t man = bits & 0x007FFFFFu;
  if ((bits & 0x7FFFFFFFu) == 0) {
    return static_cast<std::uint16_t>(sign);
  }
  if (exp >= 0x1F) {
    // Overflow -> inf.
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }
  if (exp <= 0) {
    if (exp < -10) {
      // Underflow -> zero.
      return static_cast<std::uint16_t>(sign);
    }
    // Subnormal.
    const std::uint32_t man_out = (man | 0x00800000u) >> (1 - exp);
    return static_cast<std::uint16_t>(sign | (man_out >> 13));
  }
  return static_cast<std::uint16_t>(sign | (exp << 10) | (man >> 13));
}

inline std::uint16_t Bf16BitsToFloatBits(std::uint16_t b) {
  return static_cast<std::uint16_t>(0);
}

}  // namespace detail

template <>
struct Caster<Device::Type::kKunlun> {
  template <typename Dst, typename Src>
  static Dst Cast(Src&& x) {
    static_assert(!std::is_reference_v<Dst>,
                  "`Cast` cannot return reference types");

    using PureDst = std::remove_cv_t<std::remove_reference_t<Dst>>;
    using PureSrc = std::remove_cv_t<std::remove_reference_t<Src>>;

    if constexpr (std::is_same_v<PureDst, PureSrc>) {
      return std::forward<Src>(x);
    } else {
      return FromFloatHelper<PureDst>(ToFloatHelper(std::forward<Src>(x)));
    }
  }

 private:
  template <typename T>
  static constexpr float ToFloatHelper(T&& x) {
    using PureSrc = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<PureSrc, ::infini::rt::kunlun_float16>) {
      return detail::Fp16BitsToFloat(x);
    } else {
      return static_cast<float>(std::forward<T>(x));
    }
  }

  template <typename Dst>
  static constexpr Dst FromFloatHelper(float f) {
    using PureDst = std::remove_cv_t<std::remove_reference_t<Dst>>;
    if constexpr (std::is_same_v<PureDst, ::infini::rt::kunlun_float16>) {
      return detail::FloatToFp16Bits(f);
    } else {
      return static_cast<Dst>(f);
    }
  }
};

}  // namespace infini::ops

#endif
