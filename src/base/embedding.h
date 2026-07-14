#ifndef INFINI_OPS_BASE_EMBEDDING_H_
#define INFINI_OPS_BASE_EMBEDDING_H_

#include <cstddef>

#include "data_type.h"
#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class Embedding : public Operator<Embedding> {
 public:
  Embedding(const Tensor weight, const Tensor indices,
            const int64_t padding_idx, const bool scale_grad_by_freq,
            const bool sparse, Tensor out)
      : indices_shape_{indices.shape()},
        weight_shape_{weight.shape()},
        out_shape_{out.shape()},
        indices_strides_{indices.strides()},
        weight_strides_{weight.strides()},
        out_strides_{out.strides()},
        indices_dtype_{indices.dtype()},
        weight_dtype_{weight.dtype()},
        out_dtype_{out.dtype()},
        num_indices_{NumIndices(indices_shape_)},
        vocab_size_{weight.size(0)},
        embedding_dim_{weight.size(1)},
        padding_idx_{padding_idx},
        scale_grad_by_freq_{scale_grad_by_freq},
        sparse_{sparse} {
    assert(weight.ndim() == 2 && "`Embedding` requires 2D `weight`");
    assert(out.ndim() == indices.ndim() + 1 &&
           "`Embedding` output rank must be indices rank + 1");

    for (Tensor::Size i = 0; i < indices.ndim(); ++i) {
      assert(out.size(i) == indices.size(i) &&
             "`Embedding` output shape must match `indices` on non-last "
             "dims");
    }

    assert(out.size(-1) == embedding_dim_ &&
           "`Embedding` output last dim must equal `weight` embedding dim");
    assert((indices_dtype_ == DataType::kInt32 ||
            indices_dtype_ == DataType::kInt64) &&
           "`Embedding` supports int32 and int64 indices only");
    assert((weight_dtype_ == DataType::kFloat32 ||
            weight_dtype_ == DataType::kFloat16 ||
            weight_dtype_ == DataType::kBFloat16) &&
           "`Embedding` supports float32, float16, and bfloat16 weights only");
    assert(out_dtype_ == weight_dtype_ &&
           "`Embedding` output dtype must match `weight` dtype");
    assert(padding_idx_ >= -static_cast<int64_t>(vocab_size_) &&
           padding_idx_ < static_cast<int64_t>(vocab_size_) &&
           "`Embedding` padding_idx must be within the weight rows");
  }

  Embedding(const Tensor weight, const Tensor indices, Tensor out)
      : Embedding{weight, indices, -1, false, false, out} {}

  virtual void operator()(const Tensor weight, const Tensor indices,
                          const int64_t padding_idx,
                          const bool scale_grad_by_freq, const bool sparse,
                          Tensor out) const = 0;

  void operator()(const Tensor weight, const Tensor indices, Tensor out) const {
    (*this)(weight, indices, -1, false, false, out);
  }

  template <typename TensorLike>
  static auto MakeReturnValue(const TensorLike& weight,
                              const TensorLike& indices,
                              const int64_t /*padding_idx*/ = -1,
                              const bool /*scale_grad_by_freq*/ = false,
                              const bool /*sparse*/ = false) {
    auto out_shape = indices.shape();
    out_shape.push_back(weight.size(1));

    return TensorLike::Empty(out_shape, weight.dtype(), weight.device());
  }

 protected:
  static Tensor::Size NumIndices(const Tensor::Shape& input_shape) {
    Tensor::Size num_indices = 1;

    for (Tensor::Size dim : input_shape) {
      num_indices *= dim;
    }

    return num_indices;
  }

  Tensor::Shape indices_shape_;

  Tensor::Shape weight_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides indices_strides_;

  Tensor::Strides weight_strides_;

  Tensor::Strides out_strides_;

  DataType indices_dtype_;

  DataType weight_dtype_;

  DataType out_dtype_;

  Tensor::Size num_indices_{0};

  Tensor::Size vocab_size_{0};

  Tensor::Size embedding_dim_{0};

  int64_t padding_idx_{0};

  bool scale_grad_by_freq_{false};

  bool sparse_{false};
};

}  // namespace infini::ops

#endif
