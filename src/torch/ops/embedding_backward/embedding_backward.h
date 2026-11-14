#ifndef INFINI_OPS_TORCH_EMBEDDING_BACKWARD_H_
#define INFINI_OPS_TORCH_EMBEDDING_BACKWARD_H_

#include "base/embedding_backward.h"

namespace infini::ops {

template <Device::Type kDev>
class Operator<EmbeddingBackward, kDev, 1> : public EmbeddingBackward {
 public:
  Operator(const Tensor grad_output, const Tensor indices,
           const int64_t num_weights, const int64_t padding_idx,
           const bool scale_grad_by_freq, const bool sparse,
           Tensor grad_weight);

  void operator()(const Tensor grad_output, const Tensor indices,
                  const int64_t num_weights, const int64_t padding_idx,
                  const bool scale_grad_by_freq, const bool sparse,
                  Tensor grad_weight) const override;

 private:
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_TORCH_EMBEDDING_BACKWARD_H_
