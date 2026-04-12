#ifndef INFINI_OPS_HASH_H_
#define INFINI_OPS_HASH_H_

#include <functional>
#include <vector>

template <typename T>
inline void HashCombine(std::size_t& seed, const T& v) {
  std::hash<std::decay_t<decltype(v)>> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T>
inline void HashCombine(std::size_t& seed, const std::vector<T>& v) {
  HashCombine(seed, v.size());
  for (const auto& elem : v) {
    HashCombine(seed, elem);
  }
}

#endif
