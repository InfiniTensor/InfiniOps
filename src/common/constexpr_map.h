#ifndef INFINI_OPS_COMMON_CONSTEXPR_MAP_H_
#define INFINI_OPS_COMMON_CONSTEXPR_MAP_H_

#include <array>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace infini::ops {

template <typename Key, typename Value, std::size_t N>
struct ConstexprMap {
  std::array<std::pair<Key, Value>, N> data;

  constexpr Value at(Key key) const {
    for (const auto &pr : data) {
      if (pr.first == key) return pr.second;
    }
    // TODO(lzm): change to logging
    std::cerr << "ConstexprMap's key not found at " << __FILE__ << ":"
              << __LINE__ << std::endl;
    std::abort();
  }
};

}  // namespace infini::ops

#endif
