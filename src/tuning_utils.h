#ifndef INFINI_OPS_TUNING_UTILS_H_
#define INFINI_OPS_TUNING_UTILS_H_

#include <cstdlib>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "device.h"
#include "tensor.h"

namespace infini::ops {

namespace detail {

template <typename Key>
std::string ExtractOperatorName() {
#if defined(__GNUC__) || defined(__clang__)
  std::string_view sig = __PRETTY_FUNCTION__;

  auto key_pos = sig.find("Key = ");
  if (key_pos == std::string_view::npos) return "UnknownOp";

  key_pos += 6;
  auto end_pos = sig.find_first_of("]>;", key_pos);
  std::string full_name(sig.substr(key_pos, end_pos - key_pos));

  auto last_colon = full_name.rfind("::");
  if (last_colon != std::string::npos) {
    return full_name.substr(last_colon + 2);
  }
  return full_name;
#elif defined(_MSC_VER)
  std::string_view sig = __FUNCSIG__;
  auto key_pos = sig.find("Key=");
  if (key_pos == std::string_view::npos) return "UnknownOp";
  key_pos += 4;
  auto end_pos = sig.find_first_of("]>,", key_pos);
  std::string full_name(sig.substr(key_pos, end_pos - key_pos));
  auto last_colon = full_name.rfind("::");
  if (last_colon != std::string::npos) {
    return full_name.substr(last_colon + 2);
  }
  return full_name;
#else
  return "UnknownOp";
#endif
}

inline int EnvInt(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  int parsed = std::atoi(v);
  return parsed > 0 ? parsed : fallback;
}

inline Device::Type FirstDeviceTypeHelper(bool& found) {
  found = false;
  return Device::Type::kCount;
}

template <typename First, typename... Rest>
Device::Type FirstDeviceTypeHelper(bool& found, const First& first,
                                   const Rest&... rest) {
  if constexpr (std::is_same_v<std::decay_t<First>, Tensor>) {
    found = true;
    return first.device().type();
  } else if constexpr (std::is_same_v<std::decay_t<First>,
                                      std::vector<Tensor>>) {
    if (!first.empty()) {
      found = true;
      return first.front().device().type();
    }
    return FirstDeviceTypeHelper(found, rest...);
  } else {
    return FirstDeviceTypeHelper(found, rest...);
  }
}

template <typename... Args>
Device::Type FirstDeviceType(const Args&... args) {
  bool found = false;
  return FirstDeviceTypeHelper(found, args...);
}

}  // namespace detail

}  // namespace infini::ops

#endif  // INFINI_OPS_TUNING_UTILS_H_
