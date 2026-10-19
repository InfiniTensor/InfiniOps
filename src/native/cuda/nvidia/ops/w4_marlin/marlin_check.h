#pragma once

#include <sstream>
#include <stdexcept>
#include <utility>

namespace infini::ops::native_w4 {
template <typename... Args>
[[noreturn]] inline void marlin_fail(Args&&... args) {
  std::ostringstream message;
  message << "Marlin check failed";
  if constexpr (sizeof...(args) > 0) {
    message << ": ";
    (message << ... << std::forward<Args>(args));
  }
  throw std::runtime_error(message.str());
}
}  // namespace infini::ops::native_w4

// Local replacement for the upstream checks, including constexpr factories.
#define INFINI_MARLIN_CHECK(condition, ...)               \
  do {                                                    \
    if (!(condition)) {                                   \
      ::infini::ops::native_w4::marlin_fail(__VA_ARGS__); \
    }                                                     \
  } while (false)
