#ifndef INFINI_OPS_BASE_TOPK_TOPP_SAMPLING_H_
#define INFINI_OPS_BASE_TOPK_TOPP_SAMPLING_H_

#include <cassert>
#include <cstdint>

#include "operator.h"

namespace infini::ops {

// Top-k/top-p sampling operator.
//
// Performs fused top-k and top-p filtering followed by random sampling
// from the filtered probability distribution.
//
// Input layout:
//   probs : [batch_size, vocab_size] float16/float32 — probability distribution
//           (softmax output, must sum to 1 along dim=-1).
//
// Parameters:
//   topk  : int64_t — number of highest-probability tokens to keep (0 = disabled).
//   topp  : double  — cumulative probability threshold (0.0 = disabled).
//
// Output layout:
//   out   : [batch_size] int32 — sampled token indices.
class TopkToppSampling : public Operator<TopkToppSampling> {
 public:
  TopkToppSampling(const Tensor probs, int64_t topk, double topp, Tensor out)
      : batch_size_{probs.size(0)},
        vocab_size_{probs.size(1)},
        topk_{topk},
        topp_{topp},
        dtype_{probs.dtype()} {
    assert(probs.ndim() == 2 &&
           "`TopkToppSampling` requires `probs` to be 2D [batch_size, "
           "vocab_size].");
    assert(out.ndim() == 1 &&
           "`TopkToppSampling` requires `out` to be 1D [batch_size].");
    assert(out.size(0) == probs.size(0) &&
           "`TopkToppSampling` requires `out` and `probs` to have the same "
           "batch_size.");
  }

  virtual void operator()(const Tensor probs, int64_t topk, double topp,
                          Tensor out) const = 0;

 protected:
  Tensor::Size batch_size_{0};

  Tensor::Size vocab_size_{0};

  int64_t topk_{0};

  double topp_{0.0};

  const DataType dtype_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_TOPK_TOPP_SAMPLING_H_
