#ifndef INFINI_OPS_COMMON_GENERIC_UTILS_H_
#define INFINI_OPS_COMMON_GENERIC_UTILS_H_

#include <cstddef>

namespace infini::ops::utils {

std::size_t IndexToOffset(std::size_t flat_index, std::size_t ndim,
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

// Aligned vector type for vectorized memory access.
//
// Maps (T, VEC_SIZE) to a POD type with the same size as T[VEC_SIZE] and
// natural alignment.  Used for 128-bit coalesced load/store in CUDA kernels.
template <typename T, int VEC_SIZE>
struct AlignedVec {
  using type = struct alignas(sizeof(T) * VEC_SIZE) { T data[VEC_SIZE]; };
};

// Compute the optimal vectorization factor for type T.
// Target: 128-bit (16-byte) loads where possible.
template <typename T>
constexpr int OptimalVecSize() {
  constexpr int kTargetBytes = 16;
  constexpr int vec = kTargetBytes / sizeof(T);

  return vec > 0 ? vec : 1;
}

}  // namespace infini::ops::utils

#endif
