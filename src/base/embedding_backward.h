#ifndef INFINI_OPS_BASE_EMBEDDING_BACKWARD_H_
#define INFINI_OPS_BASE_EMBEDDING_BACKWARD_H_

#include <cassert>
#include <cstdint>
#include <initializer_list>

#include "operator.h"

namespace infini::ops {

class EmbeddingBackward : public Operator<EmbeddingBackward> {
 public:
  EmbeddingBackward(const Tensor grad_output, const Tensor indices,
                    const int64_t num_weights, const int64_t padding_idx,
                    const bool scale_grad_by_freq, const bool sparse,
                    Tensor grad_weight)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        grad_weight_shape_{grad_weight.shape()},
        grad_weight_strides_{grad_weight.strides()},
        grad_weight_type_{grad_weight.dtype()},
        num_weights_{num_weights},
        padding_idx_{padding_idx},
        scale_grad_by_freq_{scale_grad_by_freq},
        sparse_{sparse},
        device_index_{grad_weight.device().index()} {
    AssertTrainingDevices({grad_output, indices, grad_weight});
    assert(!(sparse_ && scale_grad_by_freq_) &&
           "Sparse embedding does not support frequency scaling.");
    assert((indices_type_ == DataType::kInt32 ||
            indices_type_ == DataType::kInt64) &&
           "`EmbeddingBackward` requires int32 or int64 `indices`");
    assert(grad_output_type_ == grad_weight_type_ &&
           "`EmbeddingBackward` requires `grad_output` and `grad_weight` to "
           "have the same dtype");
    assert(grad_weight.ndim() == 2 &&
           "`EmbeddingBackward` requires 2D `grad_weight`");
    assert(grad_weight.size(0) == num_weights_ &&
           "`EmbeddingBackward` requires `grad_weight` row count to equal "
           "`num_weights`");
    assert(grad_output.ndim() == indices.ndim() + 1 &&
           "`EmbeddingBackward` requires `grad_output` rank to be `indices` "
           "rank + 1");
    for (Tensor::Size i = 0; i < indices.ndim(); ++i) {
      assert(grad_output.size(i) == indices.size(i) &&
             "`EmbeddingBackward` requires matching leading dims");
    }
    assert(grad_output.size(-1) == grad_weight.size(1) &&
           "`EmbeddingBackward` requires last dim of `grad_output` to equal "
           "embedding dim");
  }

  virtual void operator()(const Tensor grad_output, const Tensor indices,
                          const int64_t num_weights, const int64_t padding_idx,
                          const bool scale_grad_by_freq, const bool sparse,
                          Tensor grad_weight) const = 0;

 protected:
  static void AssertTrainingDevices(std::initializer_list<Tensor> tensors) {
    if (tensors.size() == 0) return;
    const auto device = tensors.begin()->device();
    for (const auto& tensor : tensors) {
      assert(tensor.device() == device &&
             "Training tensors must share a device type and index.");
    }
  }

  Tensor::Shape grad_output_shape_;
  Tensor::Strides grad_output_strides_;
  DataType grad_output_type_;
  Tensor::Shape indices_shape_;
  Tensor::Strides indices_strides_;
  DataType indices_type_;
  Tensor::Shape grad_weight_shape_;
  Tensor::Strides grad_weight_strides_;
  DataType grad_weight_type_;
  int64_t num_weights_{0};
  int64_t padding_idx_{-1};
  bool scale_grad_by_freq_{false};
  bool sparse_{false};
  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_EMBEDDING_BACKWARD_H_
