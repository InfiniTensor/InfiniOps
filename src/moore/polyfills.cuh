#ifndef INFINI_OPS_MOORE_POLYFILLS_CUH_
#define INFINI_OPS_MOORE_POLYFILLS_CUH_

#include <musa_bf16.h>

namespace infini::ops {

namespace detail {

template <typename T, typename = void>
struct HasHAdd : std::false_type {};

template <typename T>
struct HasHAdd<
    T, std::void_t<
           decltype(__hadd(std::declval<T>(), std::declval<T>())),
           std::enable_if_t<std::is_convertible_v<
               decltype(__hadd(std::declval<T>(), std::declval<T>())), T>>>>
    : std::true_type {};

template <typename T>
inline constexpr bool HasHAddValue = HasHAdd<T>::value;

}  // namespace detail

template <typename T>
__device__ __forceinline__ T __hadd(const T& a, const T& b) {
  if constexpr (detail::HasHAdd<T>::value) {
    return ::__hadd(a, b);
  } else {
    return a + b;
  }
}

}  // namespace infini::ops

#endif
