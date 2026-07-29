// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vLLM at commit 9b9fc4039c25a6e4fe0ae97361b62edd74b8b47e:
// csrc/core/scalar_type.hpp

#ifndef INFINI_OPS_NVIDIA_MARLIN_GEMM_SCALAR_TYPE_H_
#define INFINI_OPS_NVIDIA_MARLIN_GEMM_SCALAR_TYPE_H_

#include <cstdint>

namespace vllm {

// Marlin encodes sub-byte formats as stable integral IDs. This dependency-free
// subset preserves that encoding and only exposes the operations used by the
// private CUDA templates.
class ScalarType {
 public:
  enum NanRepr : uint8_t {
    NAN_NONE = 0,
    NAN_IEEE_754 = 1,
    NAN_EXTD_RANGE_MAX_MIN = 2,
  };

  using Id = int64_t;

  constexpr ScalarType(uint8_t exponent, uint8_t mantissa, bool signed_,
                       int32_t bias, bool finite_values_only = false,
                       NanRepr nan_repr = NAN_IEEE_754)
      : exponent(exponent),
        mantissa(mantissa),
        signed_(signed_),
        bias(bias),
        finite_values_only(finite_values_only),
        nan_repr(nan_repr) {}

  static constexpr ScalarType int_(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits - 1, true, bias);
  }

  static constexpr ScalarType uint(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits, false, bias);
  }

  static constexpr ScalarType float_IEEE754(uint8_t exponent,
                                            uint8_t mantissa) {
    return ScalarType(exponent, mantissa, true, 0, false, NAN_IEEE_754);
  }

  static constexpr ScalarType float_(uint8_t exponent, uint8_t mantissa,
                                     bool finite_values_only,
                                     NanRepr nan_repr) {
    return ScalarType(exponent, mantissa, true, 0, finite_values_only,
                      nan_repr);
  }

  constexpr Id id() const {
    auto encoded = static_cast<uint64_t>(exponent);
    encoded |= static_cast<uint64_t>(mantissa) << 8;
    encoded |= static_cast<uint64_t>(signed_) << 16;
    encoded |= static_cast<uint64_t>(static_cast<uint32_t>(bias)) << 17;
    encoded |= static_cast<uint64_t>(finite_values_only) << 49;
    encoded |= static_cast<uint64_t>(nan_repr) << 50;
    return static_cast<Id>(encoded);
  }

  static constexpr ScalarType from_id(Id id) {
    const auto encoded = static_cast<uint64_t>(id);
    return ScalarType(static_cast<uint8_t>(encoded & 0xff),
                      static_cast<uint8_t>((encoded >> 8) & 0xff),
                      static_cast<bool>((encoded >> 16) & 1),
                      static_cast<int32_t>((encoded >> 17) & 0xffffffff),
                      static_cast<bool>((encoded >> 49) & 1),
                      static_cast<NanRepr>((encoded >> 50) & 0xff));
  }

  constexpr int64_t size_bits() const {
    return mantissa + exponent + static_cast<int64_t>(signed_);
  }

  constexpr bool operator==(const ScalarType& other) const {
    return exponent == other.exponent && mantissa == other.mantissa &&
           signed_ == other.signed_ && bias == other.bias &&
           finite_values_only == other.finite_values_only &&
           nan_repr == other.nan_repr;
  }

  constexpr bool operator!=(const ScalarType& other) const {
    return !(*this == other);
  }

  const uint8_t exponent;
  const uint8_t mantissa;
  const bool signed_;
  const int32_t bias;
  const bool finite_values_only;
  const NanRepr nan_repr;
};

using ScalarTypeId = ScalarType::Id;

inline constexpr auto kS4 = ScalarType::int_(4);
inline constexpr auto kU4 = ScalarType::uint(4);
inline constexpr auto kU4B8 = ScalarType::uint(4, 8);
inline constexpr auto kS8 = ScalarType::int_(8);
inline constexpr auto kU8 = ScalarType::uint(8);
inline constexpr auto kU8B128 = ScalarType::uint(8, 128);
inline constexpr auto kFE2M1f =
    ScalarType::float_(2, 1, true, ScalarType::NAN_NONE);
inline constexpr auto kFE4M3fn =
    ScalarType::float_(4, 3, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
inline constexpr auto kFE8M0fnu =
    ScalarType(8, 0, false, 0, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
inline constexpr auto kFE8M7 = ScalarType::float_IEEE754(8, 7);
inline constexpr auto kFE5M10 = ScalarType::float_IEEE754(5, 10);
inline constexpr auto kFloat16 = kFE5M10;
inline constexpr auto kBFloat16 = kFE8M7;

}  // namespace vllm

#endif  // INFINI_OPS_NVIDIA_MARLIN_GEMM_SCALAR_TYPE_H_
