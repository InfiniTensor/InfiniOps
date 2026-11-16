#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

#include "native/cuda/ops/add/linear_layout.h"
using infini::ops::add_linear::Classify;
using infini::ops::add_linear::Layout;
using Shape = std::vector<size_t>;
using Strides = std::vector<ptrdiff_t>;
void check(Shape shape, Strides strides, bool accepted, unsigned mode = 0) {
  Layout l;
  assert(Classify(shape, strides, l) == accepted);
  if (!accepted) return;
  assert(l.mode == mode);
  size_t n = 1;
  for (auto d : shape) n *= d;
  for (size_t j = 0; j < n; ++j) {
    size_t q = j;
    ptrdiff_t ref = 0;
    for (int i = shape.size() - 1; i >= 0; --i) {
      ref += (q % shape[i]) * strides[i];
      q /= shape[i];
    }
    size_t fast = l.mode == 0   ? j
                  : l.mode == 1 ? 0
                                : (j / l.inner) % l.channels;
    assert(fast == static_cast<size_t>(ref));
  }
}
int main() {
  check({16, 64, 7, 7}, {3136, 49, 7, 1}, true, 0);
  check({16, 64, 7, 7}, {0, 1, 0, 0}, true, 2);
  check({16, 1000}, {0, 1}, true, 2);
  check({16, 64, 7, 7}, {0, 0, 0, 0}, true, 1);
  check({2, 1, 3, 4}, {12, 0, 4, 1}, true, 0);
  check({2, 1, 3, 4}, {0, 777, 1, 0}, true, 2);
  check({2, 3}, {1, 2}, false);
  check({2, 3}, {0, -1}, false);
  check({2, 3, 4}, {0, 4, 1}, false);
  check({2, 3, 4}, {0, 0, 2}, false);
  check({2, 3}, {0}, false);
  check({0, 3}, {0, 1}, false);
  check({65536, 65536}, {65536, 1}, false);
  std::cout << "13 layout cases: exact offset enumeration passed\n";
}
