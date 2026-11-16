#ifndef INFINI_OPS_ADD_LINEAR_LAYOUT_H_
#define INFINI_OPS_ADD_LINEAR_LAYOUT_H_
#include <cstdint>
#include <limits>
namespace infini::ops::add_linear {

// Host classification of the already validated, broadcast-expanded strides.
// Only dense, scalar, or one contiguous broadcast axis is admitted.
struct Layout {
  uint32_t mode = 0;

  uint32_t inner = 1;

  uint32_t channels = 1;
};

template <class Shape, class Strides>
bool Classify(const Shape& shape, const Strides& strides, Layout& result) {
  if (shape.size() != strides.size()) return false;
  uint64_t count = 1;
  for (auto n : shape) {
    if (!n || n > std::numeric_limits<uint32_t>::max() ||
        count > std::numeric_limits<uint32_t>::max() / n)
      return false;
    count *= n;
  }
  uint64_t expected = 1;
  bool dense = true;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] > 1) {
      if (strides[i] != static_cast<int64_t>(expected)) dense = false;
    }
    expected *= shape[i];
  }
  if (dense) {
    result = {0, 1, 1};
    return true;
  }
  // Count axes separately: an invalid earlier axis must never be overwritten.
  int axes = 0, axis = -1;
  for (unsigned i = 0; i < shape.size(); ++i)
    if (shape[i] > 1 && strides[i] != 0) {
      if (strides[i] != 1) return false;
      ++axes;
      axis = static_cast<int>(i);
    }
  if (axes == 0) {
    result = {1, 1, 1};
    return true;
  }
  if (axes != 1) return false;
  uint32_t inner = 1;
  for (unsigned i = axis + 1; i < shape.size(); ++i) inner *= shape[i];
  result = {2, inner, static_cast<uint32_t>(shape[axis])};
  return true;
}

}  // namespace infini::ops::add_linear
#endif
