#ifndef INFINI_OPS_IMPL_H_
#define INFINI_OPS_IMPL_H_

#include <cstddef>

namespace infini::ops {

// Global implementation index constants for the common case:
// a hand-written default and a DSL-generated alternative.
struct Impl {
  static constexpr std::size_t kDefault = 0;
  static constexpr std::size_t kDsl = 1;
};

}  // namespace infini::ops

#endif
