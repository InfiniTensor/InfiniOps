#include <cassert>
#include <initializer_list>

#include "native/cuda/ops/paged_attention_infinilm/reduction_policy.h"

int main() {
  using infini::ops::detail::CanBatchDecodeQkReduction;
  for (int batch : {1, 2}) {
    const auto accepts = [&](bool iluvatar = true, bool bf16 = true,
                             bool int32 = true, int b = 0, int heads = 32,
                             int kv = 2, int dim = 128, int blocks = 1,
                             int block = 4096, int splits = 4,
                             bool alibi = false, int stride = 4096) {
      return CanBatchDecodeQkReduction(iluvatar, bf16, int32, b ? b : batch,
                                       heads, kv, dim, blocks, block, splits,
                                       alibi, stride, 128, 4096, 128, 1048576,
                                       128, 524288, 1048576, 128, 524288);
    };
    assert(accepts());
    assert(!accepts(false));
    assert(!accepts(true, false));
    assert(!accepts(true, true, false));
    assert(!accepts(true, true, true, 3));
    assert(!accepts(true, true, true, batch, 16));
    assert(!accepts(true, true, true, batch, 32, 4));
    assert(!accepts(true, true, true, batch, 32, 2, 64));
    assert(!accepts(true, true, true, batch, 32, 2, 128, 2));
    assert(!accepts(true, true, true, batch, 32, 2, 128, 1, 2048));
    assert(!accepts(true, true, true, batch, 32, 2, 128, 1, 4096, 8));
    assert(!accepts(true, true, true, batch, 32, 2, 128, 1, 4096, 4, true));
    assert(
        !accepts(true, true, true, batch, 32, 2, 128, 1, 4096, 4, false, 4097));
  }
}
