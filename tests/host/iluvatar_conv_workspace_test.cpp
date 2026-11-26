#include <cassert>
#include <limits>
#include <string>

#include "native/cuda/iluvatar/ops/convolution/workspace.h"

int main() {
  using infini::ops::iluvatar_conv_detail::WorkspaceLimit;
  const auto default_limit = WorkspaceLimit("");
  const auto zero_limit = WorkspaceLimit("0");
  const auto expanded_limit = WorkspaceLimit("256");
  assert(default_limit && *default_limit == 64 * 1024 * 1024);
  assert(zero_limit && *zero_limit == 0);
  assert(expanded_limit && *expanded_limit == 256 * 1024 * 1024);
  for (const auto* invalid : {"-1", "2GB", "256 ", " 256", "+256"}) {
    const auto result = WorkspaceLimit(invalid);
    assert(!result);
  }
  const auto overflow =
      WorkspaceLimit(std::to_string(std::numeric_limits<size_t>::max()));
  assert(!overflow);
}
