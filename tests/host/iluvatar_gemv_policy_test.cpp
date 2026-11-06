#include <cassert>
#include <initializer_list>
#include <limits>

#include "native/cuda/iluvatar/ops/gemm/gemv_policy.h"

int main() {
  using infini::ops::iluvatar::CanUseSmallBatchGemv;
  for (int down : {0, 1}) {
    const int k = down ? 16384 : 4096;
    const int n = down ? 4096 : 32768;
    for (int m : {1, 2}) {
      const auto valid = [&](bool bf16 = true, bool same = true,
                             bool bias = false, float alpha = 1, float beta = 0,
                             int ta = 0, int tb = 0, int rows = 0,
                             int a_stride = 0, std::uintptr_t address = 4096) {
        return CanUseSmallBatchGemv(
            bf16, same, bias, alpha, beta, ta, tb, rows ? rows : m, k, n,
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
    }
  }
}
