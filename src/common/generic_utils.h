#ifndef INFINI_OPS_COMMON_GENERIC_UTILS_H_
#define INFINI_OPS_COMMON_GENERIC_UTILS_H_

#include <cstddef>

namespace infini::ops::utils {

inline std::size_t IndexToOffset(std::size_t flat_index, std::size_t ndim,
                                 const std::size_t* shape,
                                 const std::ptrdiff_t* strides) {
  std::size_t res = 0;
  for (std::size_t i = ndim; i-- > 0;) {
    res += (flat_index % shape[i]) * strides[i];
    flat_index /= shape[i];
  }
  return res;
}

template <typename X, typename Y>
constexpr auto CeilDiv(const X& x, const Y& y) {
  return (x + y - 1) / y;
}

// log2 of x when x is a power of two, else -1.
template <typename X>
constexpr int Log2IfPowerOfTwo(const X& x) {
  if (x <= 0 || (x & (x - 1)) != 0) return -1;
  int shift = 0;
  for (X bit = 1; bit != x; bit <<= 1) ++shift;
  return shift;
}

}  // namespace infini::ops::utils

#endif
