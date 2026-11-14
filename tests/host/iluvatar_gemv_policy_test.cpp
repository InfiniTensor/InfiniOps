#include <cassert>
#include <initializer_list>
#include <limits>
#include <utility>

#include "native/cuda/iluvatar/ops/gemm/gemv_policy.h"

int main() {
  using infini::ops::iluvatar::CanUseSmallBatchGemv;
  for (const auto [k, n] : {std::pair{2048, 4096},
                            {4160, 2052},
                            {4224, 2056},
                            {4288, 2060},
                            {2048, 11008},
                            {5504, 2048},
                            {3072, 16384},
                            {8192, 3072},
                            {4096, 22016},
                            {11008, 4096},
                            {4096, 28672},
                            {14336, 4096},
                            {5120, 27648},
                            {13824, 5120},
                            {8192, 57344},
                            {28672, 8192},
                            {4096, 32768},
                            {16384, 4096}}) {
    for (int m : {1, 2}) {
      const auto valid = [&](bool all_bf16 = true, bool same = true,
                             bool bias = false, float alpha = 1, float beta = 0,
                             int ta = 0, int tb = 0, int rows = 0,
                             int a_stride = 0, std::uintptr_t address = 4096) {
        return CanUseSmallBatchGemv(
            all_bf16, same, bias, alpha, beta, ta, tb, rows ? rows : m, k, n,
            a_stride ? a_stride : k, 1, 1, k, n, 1, address, 8192);
      };
      assert(valid());
      assert(!valid(false));
      assert(!valid(true, false));
      assert(!valid(true, true, true));
      assert(!valid(true, true, false, 0.5f));
      assert(!valid(true, true, false, 1, 1));
      assert(!valid(true, true, false, 1, 0, 1));
      assert(!valid(true, true, false, 1, 0, 0, 1));
      assert(!valid(true, true, false, 1, 0, 0, 0, 3));
      assert(!valid(true, true, false, 1, 0, 0, 0, m, k + 1));
      assert(!valid(true, true, false, 1, 0, 0, 0, m, k, 4098));
      assert(
          !valid(true, true, false, std::numeric_limits<float>::quiet_NaN()));
      assert(!CanUseSmallBatchGemv(true, true, false, 1, 0, 0, 0, m, k, n + 1,
                                   k, 1, 1, k, n + 1, 1, 4096, 8192));
      assert(!CanUseSmallBatchGemv(true, true, false, 1, 0, 0, 0, m, k, n, k, 1,
                                   n, 1, n, 1, 4096, 8192));
      assert(!CanUseSmallBatchGemv(true, true, false, 1, 0, 0, 0, m, k, n, k, 1,
                                   1, k, n, 1, 4096, 8194));
      assert(!CanUseSmallBatchGemv(true, true, false, 1, 0, 0, 0, m, k, n, k, 1,
                                   1, k, n + 1, 1, 4096, 8192));
    }
  }
  for (std::size_t m : {0, 3, 4}) {
    assert(!CanUseSmallBatchGemv(true, true, false, 1, 0, 0, 0, m, 2048, 4096,
                                 2048, 1, 1, 2048, 4096, 1, 4096, 8192));
  }
  const std::size_t overflow =
      static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
  for (std::size_t k :
       {std::size_t{0}, std::size_t{768}, std::size_t{1025}, overflow}) {
    assert(!CanUseSmallBatchGemv(true, true, false, 1, 0, 0, 0, 1, k, 16384, k,
                                 1, 1, k, 16384, 1, 4096, 8192));
  }
  for (std::size_t n :
       {std::size_t{0}, std::size_t{1020}, std::size_t{1025}, overflow}) {
    assert(!CanUseSmallBatchGemv(true, true, false, 1, 0, 0, 0, 1, 16384, n,
                                 16384, 1, 1, 16384, n, 1, 4096, 8192));
  }
  // Weights below the measured size threshold retain the BLAS implementation.
  assert(!CanUseSmallBatchGemv(true, true, false, 1, 0, 0, 0, 1, 2048, 4092,
                               2048, 1, 1, 2048, 4092, 1, 4096, 8192));
}
