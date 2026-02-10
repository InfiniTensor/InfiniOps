#ifndef INFINI_OPS_DATA_TYPE_H_
#define INFINI_OPS_DATA_TYPE_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace infini::ops {

class DataType {
 public:
  constexpr DataType(int index, std::size_t element_size, const char* name)
      : index_{index}, element_size_{element_size}, name_{name} {}

  static const DataType& FromString(const std::string& name);

  constexpr bool operator==(const DataType& other) const {
    return index_ == other.index_;
  }

  constexpr std::size_t element_size() const { return element_size_; }

  constexpr const char* name() const { return name_; }

 private:
  int index_{0};

  std::size_t element_size_{0};

  const char* name_{nullptr};
};

constexpr DataType kInt8{0, sizeof(int8_t), "int8"};

constexpr DataType kInt16{1, sizeof(int16_t), "int16"};

constexpr DataType kInt32{2, sizeof(int32_t), "int32"};

constexpr DataType kInt64{3, sizeof(int64_t), "int64"};

constexpr DataType kUInt8{4, sizeof(uint8_t), "uint8"};

constexpr DataType kUInt16{5, sizeof(uint16_t), "uint16"};

constexpr DataType kUInt32{6, sizeof(uint32_t), "uint32"};

constexpr DataType kUInt64{7, sizeof(uint64_t), "uint64"};

constexpr DataType kFloat16{8, 2, "float16"};

constexpr DataType kBFloat16{9, 2, "bfloat16"};

constexpr DataType kFloat32{10, sizeof(float), "float32"};

constexpr DataType kFloat64{11, sizeof(double), "float64"};

inline const DataType& DataType::FromString(const std::string& name) {
  static std::unordered_map<std::string, const DataType&> name_to_dtype{
      {kInt8.name(), kInt8},       {kInt16.name(), kInt16},
      {kInt32.name(), kInt32},     {kInt64.name(), kInt64},
      {kUInt8.name(), kUInt8},     {kUInt16.name(), kUInt16},
      {kUInt32.name(), kUInt32},   {kUInt64.name(), kUInt64},
      {kFloat16.name(), kFloat16}, {kBFloat16.name(), kBFloat16},
      {kFloat32.name(), kFloat32}, {kFloat64.name(), kFloat64}};

  return name_to_dtype.at(name);
}

}  // namespace infini::ops

#endif
