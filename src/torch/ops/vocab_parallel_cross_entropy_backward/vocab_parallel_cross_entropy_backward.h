#ifndef INFINI_OPS_TORCH_VOCAB_PARALLEL_CROSS_ENTROPY_BACKWARD_H_
#define INFINI_OPS_TORCH_VOCAB_PARALLEL_CROSS_ENTROPY_BACKWARD_H_

#include "base/vocab_parallel_cross_entropy_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<VocabParallelCrossEntropyBackward, kDev, 1>
    : public VocabParallelCrossEntropyBackward {
 public:
  Operator(const Tensor grad_output, const Tensor softmax_local,
           const Tensor target_mask, const Tensor masked_target,
           const Tensor valid_mask_local, const int64_t vocab_size_local,
           const int64_t vocab_size_original, const double label_smoothing,
           Tensor grad_input);

  void operator()(const Tensor grad_output, const Tensor softmax_local,
                  const Tensor target_mask, const Tensor masked_target,
                  const Tensor valid_mask_local, const int64_t vocab_size_local,
                  const int64_t vocab_size_original,
                  const double label_smoothing,
                  Tensor grad_input) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_VOCAB_PARALLEL_CROSS_ENTROPY_BACKWARD_H_
