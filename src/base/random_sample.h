#ifndef INFINI_OPS_BASE_RANDOM_SAMPLE_H_
#define INFINI_OPS_BASE_RANDOM_SAMPLE_H_

#include "operator.h"

namespace infini::ops {

// Sample one index from a 1D `logits` vector into a scalar `out` tensor.
// The sampling policy is the same as `infinicore::op::random_sample_`:
//   - `temperature == 0` or `topk <= 1`: greedy `argmax`.
//   - Otherwise: apply temperature, softmax, truncate to the top `topk`
//     entries, then to the smallest prefix whose cumulative probability
//     reaches `topp`, renormalize, and pick the entry at cumulative
//     probability `random_val` in `[0, 1)`.
//
// Expected shapes:
//   `logits`: 1D `[vocab_size]` floating-point dtype.
//   `out`:    0D integer dtype.
class RandomSample : public Operator<RandomSample> {
 public:
  RandomSample(const Tensor logits, float random_val, float topp, int topk,
               float temperature, Tensor out)
      : logits_type_{logits.dtype()},
        out_type_{out.dtype()},
        logits_shape_{logits.shape()},
        out_shape_{out.shape()},
        logits_strides_{logits.strides()},
        out_strides_{out.strides()},
        random_val_{random_val},
        topp_{topp},
        topk_{topk},
        temperature_{temperature} {
    assert(logits.ndim() == 1 && "`logits` must be 1D");
    assert(out.ndim() == 0 && "`out` must be a 0D scalar");
  }

  virtual void operator()(const Tensor logits, float random_val, float topp,
                          int topk, float temperature,
                          Tensor out) const = 0;

 protected:
  const DataType logits_type_;

  const DataType out_type_;

  Tensor::Shape logits_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides logits_strides_;

  Tensor::Strides out_strides_;

  float random_val_{0.0f};

  float topp_{1.0f};

  int topk_{1};

  float temperature_{1.0f};
};

}  // namespace infini::ops

#endif
