#ifndef INFINI_OPS_DATA_TYPE_H_
#define INFINI_OPS_DATA_TYPE_H_

#include <cstdint>
#include <string>

#include "common/constexpr_map.h"
#include "common/traits.h"

namespace infini::ops {

enum class DataType : int8_t {
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

template <DataType DType>
struct TypeMap;

template <DataType DType>
using TypeMap_t = typename TypeMap<DType>::type;

template <typename T>
struct DataTypeMap;

template <typename T>
inline constexpr DataType DataTypeMap_v = DataTypeMap<T>::value;

#define DEFINE_DATA_TYPE_MAPPING(ENUM_VALUE, CPP_TYPE)      \
  template <>                                               \
  struct TypeMap<DataType::ENUM_VALUE> {                    \
    using type = CPP_TYPE;                                  \
  };                                                        \
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
// TODO(lzm): support fp16 and bf16

// Defines the common categories of data types using List
using FloatingTypes = List<DataType::kFloat32, DataType::kFloat64>;
using ReducedFloatingTypes = List<DataType::kFloat16, DataType::kBFloat16>;
using SignedIntegralTypes =
    List<DataType::kInt8, DataType::kInt16, DataType::kInt32, DataType::kInt64>;
using UnsignedIntegralTypes = List<DataType::kUInt8, DataType::kUInt16,
                                   DataType::kUInt32, DataType::kUInt64>;

using BitTypes8 = List<DataType::kInt8, DataType::kUInt8>;
using BitTypes16 = List<DataType::kInt16, DataType::kUInt16, DataType::kFloat16,
                        DataType::kBFloat16>;
using BitTypes32 =
    List<DataType::kInt32, DataType::kUInt32, DataType::kFloat32>;
using BitTypes64 =
    List<DataType::kInt64, DataType::kUInt64, DataType::kFloat64>;

using AllFloatingTypes = Concat_t<FloatingTypes, ReducedFloatingTypes>;
using AllIntegralTypes = Concat_t<SignedIntegralTypes, UnsignedIntegralTypes>;
using AllTypes = Concat_t<AllFloatingTypes, AllIntegralTypes>;

}  // namespace infini::ops

#endif
