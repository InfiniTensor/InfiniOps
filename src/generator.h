#ifndef INFINI_OPS_GENERATOR_H_
#define INFINI_OPS_GENERATOR_H_

#include <any>
#include <cstddef>
#include <functional>
#include <utility>

namespace infini::ops {

class Generator {
 public:
  Generator() = default;

  template <typename T>
  static Generator From(T value) {
    return Generator{std::move(value)};
  }

  template <typename T>
  const T* GetIf() const {
    return std::any_cast<T>(&value_);
  }

  // Generator state is a call-time input and does not affect construction.
  std::size_t Hash() const { return value_.type().hash_code(); }

  friend bool operator==(const Generator& lhs, const Generator& rhs) {
    return lhs.value_.type() == rhs.value_.type();
  }

 private:
  explicit Generator(std::any value) : value_{std::move(value)} {}

  std::any value_;
};

}  // namespace infini::ops

template <>
struct std::hash<infini::ops::Generator> {
  std::size_t operator()(const infini::ops::Generator& generator) const {
    return generator.Hash();
  }
};

#endif
