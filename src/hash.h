#ifndef INFINI_OPS_HASH_H_
#define INFINI_OPS_HASH_H_

#include <functional>

template <typename T>
inline void hash_combine(std::size_t& seed, const T& v) {
  std::hash<std::decay_t<decltype(v)>> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

#endif
