#ifndef INFINI_OPS_BASE_EMBEDDING_H_
#define INFINI_OPS_BASE_EMBEDDING_H_

#include <cassert>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {

// Embedding performs a token embedding lookup.
//
// Interface follows the inference-time vLLM/PyTorch convention:
//   `out = weight[input_ids]`.
//
// The input layout is:
//   `input_ids`: Any shape, `int32` or `int64`.
//   `weight`: `[vocab_size, hidden_size]`.
//   `out`: `input_ids.shape + [hidden_size]`.
class Embedding : public Operator<Embedding> {
 public:
  Embedding(const Tensor input_ids, const Tensor weight, Tensor out)
      : num_tokens_{input_ids.numel()},
        vocab_size_{weight.size(0)},
        hidden_size_{weight.size(1)},
        input_dtype_{input_ids.dtype()},
        weight_dtype_{weight.dtype()} {
    assert((input_dtype_ == DataType::kInt32 ||
            input_dtype_ == DataType::kInt64) &&
           "`Embedding` requires `input_ids` to be `int32` or `int64`.");
    assert(
        weight.ndim() == 2 &&
        "`Embedding` requires `weight` to be 2D `[vocab_size, hidden_size]`.");
    assert(out.dtype() == weight.dtype() &&
           "`Embedding` requires `out` and `weight` to have the same dtype.");
    assert(out.ndim() == input_ids.ndim() + 1 &&
           "`Embedding` requires `out.ndim == input_ids.ndim + 1`.");
    assert(out.size(-1) == hidden_size_ &&
           "`Embedding` requires `out.shape[-1] == weight.shape[-1]`.");

    for (std::size_t i = 0; i < input_ids.ndim(); ++i) {
      assert(out.size(i) == input_ids.size(i) &&
             "`Embedding` requires `out` prefix shape to match `input_ids`.");
    }
  }

  virtual void operator()(const Tensor input_ids, const Tensor weight,
                          Tensor out) const = 0;

 protected:
  Tensor::Size num_tokens_{0};

  Tensor::Size vocab_size_{0};

  Tensor::Size hidden_size_{0};

  const DataType input_dtype_;

  const DataType weight_dtype_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_EMBEDDING_H_
