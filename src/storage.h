#ifndef INFINI_OPS_STORAGE_H_
#define INFINI_OPS_STORAGE_H_

#include <any>
#include <cstddef>
#include <functional>
#include <utility>

namespace infini::ops {

class Storage {
 public:
  Storage() = default;

  template <typename T>
  static Storage From(T value) {
    return Storage{std::move(value)};
  }

  template <typename T>
  const T* GetIf() const {
    return std::any_cast<T>(&value_);
  }

  // Storage identity is a call-time input and does not affect construction.
  std::size_t Hash() const { return value_.type().hash_code(); }

  friend bool operator==(const Storage& lhs, const Storage& rhs) {
    return lhs.value_.type() == rhs.value_.type();
  }

 private:
  explicit Storage(std::any value) : value_{std::move(value)} {}

  std::any value_;
};

}  // namespace infini::ops

template <>
struct std::hash<infini::ops::Storage> {
  std::size_t operator()(const infini::ops::Storage& storage) const {
    return storage.Hash();
  }
};

#endif
