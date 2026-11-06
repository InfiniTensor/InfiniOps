#include <cassert>
#include <initializer_list>
#include <limits>

#include "native/cuda/iluvatar/ops/causal_softmax_infinilm/policy.h"

int main() {
  using infini::ops::iluvatar::CanUseCausalSoftmaxFp32;
  for (std::size_t q : {2, 521, 2051, 2910, 4096}) {
    assert(CanUseCausalSoftmaxFp32(true, true, true, 32, q, q));
    assert(CanUseCausalSoftmaxFp32(true, true, true, 64, q, 4096));
    assert(!CanUseCausalSoftmaxFp32(false, true, true, 32, q, q));
    assert(!CanUseCausalSoftmaxFp32(true, false, true, 32, q, q));
    assert(!CanUseCausalSoftmaxFp32(true, true, false, 32, q, q));
    assert(!CanUseCausalSoftmaxFp32(true, true, true, 0, q, q));
    assert(!CanUseCausalSoftmaxFp32(true, true, true, 32, q, q - 1));
    assert(!CanUseCausalSoftmaxFp32(
        true, true, true, std::numeric_limits<std::size_t>::max(), q, q));
  }
  assert(!CanUseCausalSoftmaxFp32(true, true, true, 32, 1, 4096));
  assert(!CanUseCausalSoftmaxFp32(true, true, true, 32, 0, 4096));
  assert(!CanUseCausalSoftmaxFp32(true, true, true, 32, 2051, 4097));
}
