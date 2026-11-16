#ifndef INFINI_OPS_ILUVATAR_CONVOLUTION_WORKSPACE_H_
#define INFINI_OPS_ILUVATAR_CONVOLUTION_WORKSPACE_H_

#include <charconv>
#include <cstddef>
#include <limits>
#include <optional>
#include <string_view>

namespace infini::ops::iluvatar_conv_detail {

inline std::optional<size_t> WorkspaceLimit(std::string_view text) {
  constexpr size_t unit = 1024 * 1024;
  if (text.empty()) return 64 * unit;
  size_t megabytes = 0;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), megabytes);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      megabytes > std::numeric_limits<size_t>::max() / unit)
    return std::nullopt;
  return megabytes * unit;
}

}  // namespace infini::ops::iluvatar_conv_detail

#endif
