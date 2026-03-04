#ifndef INFINI_OPS_DATA_TYPE_H_
#define INFINI_OPS_DATA_TYPE_H_

#include <cstdint>
#include <cstring>
#include <string>

#ifdef WITH_NVIDIA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#elif defined(WITH_ILUVATAR)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#elif defined(WITH_METAX)
#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>
#endif

#include "common/constexpr_map.h"
#include "common/traits.h"

namespace infini::ops {

enum class DataType : std::int8_t {
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kUInt8,
  kUInt16,
  kUInt32,
  kUInt64,
  kFloat16,
  kBFloat16,
  kFloat32,
  kFloat64
};

constexpr ConstexprMap<DataType, std::size_t, 12> kDataTypeToSize{{{
    {DataType::kInt8, 1},
    {DataType::kInt16, 2},
    {DataType::kInt32, 4},
    {DataType::kInt64, 8},
    {DataType::kUInt8, 1},
    {DataType::kUInt16, 2},
    {DataType::kUInt32, 4},
    {DataType::kUInt64, 8},
    {DataType::kFloat16, 2},
    {DataType::kBFloat16, 2},
    {DataType::kFloat32, 4},
    {DataType::kFloat64, 8},
}}};

constexpr ConstexprMap<DataType, std::string_view, 12> kDataTypeToDesc{{{
    {DataType::kInt8, "int8"},
    {DataType::kInt16, "int16"},
    {DataType::kInt32, "int32"},
    {DataType::kInt64, "int64"},
    {DataType::kUInt8, "uint8"},
    {DataType::kUInt16, "uint16"},
    {DataType::kUInt32, "uint32"},
    {DataType::kUInt64, "uint64"},
    {DataType::kFloat16, "float16"},
    {DataType::kBFloat16, "bfloat16"},
    {DataType::kFloat32, "float32"},
    {DataType::kFloat64, "float64"},
}}};

constexpr ConstexprMap<std::string_view, DataType, 12> kStringToDataType{{{
    {"int8", DataType::kInt8},
    {"int16", DataType::kInt16},
    {"int32", DataType::kInt32},
    {"int64", DataType::kInt64},
    {"uint8", DataType::kUInt8},
    {"uint16", DataType::kUInt16},
    {"uint32", DataType::kUInt32},
    {"uint64", DataType::kUInt64},
    {"float16", DataType::kFloat16},
    {"bfloat16", DataType::kBFloat16},
    {"float32", DataType::kFloat32},
    {"float64", DataType::kFloat64},
}}};

struct float16_t {
  uint16_t bits;

  static inline float16_t FromFloat(float val) {
    uint32_t f32;
    std::memcpy(&f32, &val, sizeof(f32));
    uint16_t sign = (f32 >> 16) & 0x8000;
    int32_t exponent = ((f32 >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32 & 0x7FFFFF;

    if (exponent >= 16) {
      // NaN
      if (exponent == 128 && mantissa != 0) {
        return {static_cast<uint16_t>(sign | 0x7E00)};
      }
      // Inf
      return {static_cast<uint16_t>(sign | 0x7C00)};
    } else if (exponent >= -14) {
      return {static_cast<uint16_t>(sign | ((exponent + 15) << 10) |
                                    (mantissa >> 13))};
    } else if (exponent >= -24) {
      mantissa |= 0x800000;
      mantissa >>= (-14 - exponent);
      return {static_cast<uint16_t>(sign | (mantissa >> 13))};
    }
    // Too small for subnormal: return signed zero
    return {sign};
  }

  // Conversion back to float
  inline float ToFloat() const {
    uint32_t sign = (bits & 0x8000) << 16;
    int32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;
    uint32_t f32_bits;

    if (exponent == 31) {
      f32_bits = sign | 0x7F800000 | (mantissa << 13);
    } else if (exponent == 0) {
      if (mantissa == 0) {
        f32_bits = sign;
      } else {
        exponent = -14;
        while ((mantissa & 0x400) == 0) {
          mantissa <<= 1;
          exponent--;
        }
        mantissa &= 0x3FF;
        f32_bits = sign | ((exponent + 127) << 23) | (mantissa << 13);
      }
    } else {
      f32_bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &f32_bits, sizeof(result));
    return result;
  }
};

struct bfloat16_t {
  uint16_t bits;

  static inline bfloat16_t FromFloat(float val) {
    uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));

    const uint32_t rounding_bias = 0x00007FFF + ((bits32 >> 16) & 1);
    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);
    return {bf16_bits};
  }

  inline float ToFloat() const {
    uint32_t bits32 = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &bits32, sizeof(result));
    return result;
  }
};

template <DataType dtype>
struct TypeMap;

template <DataType dtype>
using TypeMapType = typename TypeMap<dtype>::type;

template <typename T>
struct DataTypeMap;

template <typename T>
inline constexpr DataType DataTypeMapValue = DataTypeMap<T>::value;

#define DEFINE_DATA_TYPE_MAPPING(ENUM_VALUE, CPP_TYPE)      \
  template <>                                               \
  struct TypeMap<DataType::ENUM_VALUE> {                    \
    using type = CPP_TYPE;                                  \
  };                                                        \
                                                            \
  template <>                                               \
  struct DataTypeMap<CPP_TYPE> {                            \
    static constexpr DataType value = DataType::ENUM_VALUE; \
  };

DEFINE_DATA_TYPE_MAPPING(kUInt8, uint8_t)
DEFINE_DATA_TYPE_MAPPING(kInt8, int8_t)
DEFINE_DATA_TYPE_MAPPING(kUInt16, uint16_t)
DEFINE_DATA_TYPE_MAPPING(kInt16, int16_t)
DEFINE_DATA_TYPE_MAPPING(kUInt32, uint32_t)
DEFINE_DATA_TYPE_MAPPING(kInt32, int32_t)
DEFINE_DATA_TYPE_MAPPING(kUInt64, uint64_t)
DEFINE_DATA_TYPE_MAPPING(kInt64, int64_t)
DEFINE_DATA_TYPE_MAPPING(kFloat32, float)
DEFINE_DATA_TYPE_MAPPING(kFloat64, double)

#if defined(WITH_NVIDIA) || defined(WITH_ILUVATAR)
DEFINE_DATA_TYPE_MAPPING(kFloat16, half)
DEFINE_DATA_TYPE_MAPPING(kBFloat16, __nv_bfloat16)
#elif defined(WITH_METAX)
DEFINE_DATA_TYPE_MAPPING(kFloat16, __half)
DEFINE_DATA_TYPE_MAPPING(kBFloat16, __maca_bfloat16)
#else
DEFINE_DATA_TYPE_MAPPING(kFloat16, float16_t)
DEFINE_DATA_TYPE_MAPPING(kBFloat16, bfloat16_t)
#endif
#undef DEFINE_DATA_TYPE_MAPPING

// Define the traits to check whether a type is bfloat16 or float16.
template <typename T>
inline constexpr bool IsBFloat16 = (DataTypeMapValue<T> == DataType::kBFloat16);

template <typename T>
inline constexpr bool IsFP16 = (DataTypeMapValue<T> == DataType::kFloat16);

// Defines the common categories of data types using List.
using FloatTypes = List<DataType::kFloat32, DataType::kFloat64>;
using ReducedFloatTypes = List<DataType::kFloat16, DataType::kBFloat16>;
using IntTypes =
    List<DataType::kInt8, DataType::kInt16, DataType::kInt32, DataType::kInt64>;
using UIntTypes = List<DataType::kUInt8, DataType::kUInt16, DataType::kUInt32,
                       DataType::kUInt64>;

using BitTypes8 = List<DataType::kInt8, DataType::kUInt8>;
using BitTypes16 = List<DataType::kInt16, DataType::kUInt16, DataType::kFloat16,
                        DataType::kBFloat16>;
using BitTypes32 =
    List<DataType::kInt32, DataType::kUInt32, DataType::kFloat32>;
using BitTypes64 =
    List<DataType::kInt64, DataType::kUInt64, DataType::kFloat64>;

using AllFloatTypes = ConcatType<FloatTypes, ReducedFloatTypes>;
using AllIntTypes = ConcatType<IntTypes, UIntTypes>;
using AllTypes = ConcatType<AllFloatTypes, AllIntTypes>;

}  // namespace infini::ops

#endif
