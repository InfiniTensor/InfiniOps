#ifndef INFINI_OPS_TUNING_SIGNATURE_H_
#define INFINI_OPS_TUNING_SIGNATURE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#include "data_type.h"
#include "tensor.h"

namespace infini::ops {

// 调优签名：通用地从算子参数中提取形状和类型信息，用于查找最优实现。
// 设计原则：不依赖任何具体算子的实现，通过模板折叠表达式遍历参数列表。
//
// 参数分类处理：
//   - Tensor / optional<Tensor>  → 记录 shape + dtype（不含 strides）
//   - vector<Tensor>             → 展开后逐一记录
//   - 算术类型 / 枚举类型         → 记录为 double 标量
//   - optional<算术/枚举>        → 有值时记录标量，无值时跳过
//   - 其他类型（string、strides） → 跳过，不影响签名
struct TuningSignature {
  // 单个张量的签名：形状 + 数据类型（不含 strides，降低匹配粒度）
  struct TensorSig {
    std::vector<int64_t> shape;
    DataType dtype;

    bool operator==(const TensorSig& other) const {
      return shape == other.shape && dtype == other.dtype;
    }
  };

  std::vector<TensorSig> tensors;  // 所有张量参数的签名
  std::vector<double> scalars;     // 所有标量参数（如 eps）

  // 从任意参数列表构建签名（通用接口）
  template <typename... Args>
  static TuningSignature Build(const Args&... args) {
    TuningSignature sig;
    // C++17 折叠表达式：对每个参数调用 Absorb
    (sig.Absorb(args), ...);
    return sig;
  }

  // 结构化相等比较
  bool operator==(const TuningSignature& other) const {
    return tensors == other.tensors && scalars == other.scalars;
  }

  // 哈希函数（用于 unordered_map 的键）
  std::size_t Hash() const {
    std::size_t h = 0;
    for (const auto& t : tensors) {
      for (auto dim : t.shape) {
        h ^= std::hash<int64_t>{}(dim) + 0x9e3779b9 + (h << 6) + (h >> 2);
      }
      h ^= std::hash<int>{}(static_cast<int>(t.dtype)) + 0x9e3779b9 +
           (h << 6) + (h >> 2);
    }
    for (auto s : scalars) {
      h ^= std::hash<double>{}(s) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }

 private:
  // 吸收单个张量：提取 shape 和 dtype
  void Absorb(const Tensor& t) {
    std::vector<int64_t> shape_vec;
    for (std::size_t i = 0; i < t.shape().size(); ++i) {
      shape_vec.push_back(static_cast<int64_t>(t.shape()[i]));
    }
    tensors.push_back({shape_vec, t.dtype()});
  }

  // 吸收可选张量：有值则记录，无值则跳过
  void Absorb(const std::optional<Tensor>& t) {
    if (t.has_value()) {
      Absorb(*t);
    }
  }

  // 吸收张量数组
  void Absorb(const std::vector<Tensor>& ts) {
    for (const auto& t : ts) {
      Absorb(t);
    }
  }

  // 通用模板：用 if constexpr 分流，避免对不可转换的类型产生错误
  template <typename T>
  void Absorb(const T& v) {
    if constexpr (std::is_arithmetic_v<T>) {
      // 算术类型（int, float, double, bool 等）→ 记录为标量
      scalars.push_back(static_cast<double>(v));
    } else if constexpr (std::is_enum_v<T>) {
      // 枚举类型（DataType 等）→ 转 int64_t 再存为标量
      scalars.push_back(static_cast<double>(static_cast<int64_t>(v)));
    }
    // 其他类型（std::string、std::vector<Stride> 等）→ 跳过
  }

  // 吸收可选标量或枚举：有值则递归处理，无值则跳过
  template <typename T>
  void Absorb(const std::optional<T>& v) {
    if (v.has_value()) {
      Absorb(*v);
    }
  }
};

}  // namespace infini::ops

// 为 TuningSignature 提供标准哈希支持（用于 unordered_map 的键）
namespace std {
template <>
struct hash<infini::ops::TuningSignature> {
  std::size_t operator()(const infini::ops::TuningSignature& sig) const {
    return sig.Hash();
  }
};
}  // namespace std

#endif