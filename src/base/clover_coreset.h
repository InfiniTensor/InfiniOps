#pragma once

#include "operator.h"

namespace infini::ops {

// Block coreset of the Clover router. For every block of `block_size` rows it
// aggregates the softmax probabilities of the top-8 experts of each row and
// keeps, most massive first, the experts that carry `top_p` of that mass; the
// logits of the other experts become -inf. `logits` and `out` are contiguous
// `[tokens, 256]`. A provider allocates its workspace when the caller provides
// none.
class CloverCoreset : public Operator<CloverCoreset> {
 public:
  CloverCoreset(const Tensor logits, size_t block_size, double top_p,
                Tensor out) {
    assert(logits.ndim() == 2 && logits.size(1) == 256 && block_size > 0 &&
           block_size <= 32 && logits.size(0) % block_size == 0 &&
           out.shape() == logits.shape() && out.dtype() == logits.dtype() &&
           logits.IsContiguous() && out.IsContiguous() &&
           logits.device() == out.device() && top_p > 0 && top_p <= 1 &&
           "CloverCoreset: invalid logits, block size or routing mass.");
  }

  virtual void operator()(const Tensor logits, size_t block_size, double top_p,
                          Tensor out) const = 0;
};

}  // namespace infini::ops
