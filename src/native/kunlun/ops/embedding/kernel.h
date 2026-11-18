#ifndef INFINI_OPS_KUNLUN_EMBEDDING_KERNEL_H_
#define INFINI_OPS_KUNLUN_EMBEDDING_KERNEL_H_

#include <stdexcept>

#include "base/embedding.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/launch2_declarations.h"  // KunlunDtypeTag
#include "native/kunlun/runtime_.h"

namespace infini::ops {

// Kunlun embedding: flat gather of weight rows. Indices int32/int64;
// weights f32/f16/bf16. padding_idx needs no special handling at inference
// (its row is copied through); max_norm renorm is not supported.
template <>
class Operator<Embedding, Device::Type::kKunlun> : public Embedding {
 public:
  using Embedding::Embedding;

  bool capture_safe() const override { return !max_norm_.has_value(); }

  static bool supports(const Tensor input, const Tensor weight,
                       const std::optional<int64_t> padding_idx,
                       const std::optional<double> max_norm,
                       const double norm_type, const bool scale_grad_by_freq,
                       const bool sparse, const Tensor out) {
    (void)padding_idx;
    (void)norm_type;
    (void)scale_grad_by_freq;
    (void)sparse;
    (void)weight;
    if (max_norm.has_value()) return false;
    if (input.ndim() >= 1 && input.strides().back() != 1) return false;
    // Weight gather assumes row-major [vocab, dim]: stride(-2) == dim.
    if (weight.size(0) > 1 && weight.strides()[weight.strides().size() - 2] !=
                                  static_cast<Tensor::Stride>(weight.size(1))) {
      return false;
    }
    if (out.strides().back() != 1) return false;
    if (out.ndim() >= 2 && out.strides()[out.strides().size() - 2] !=
                               static_cast<Tensor::Stride>(weight.size(1))) {
      return false;
    }
    return true;
  }

  void operator()(const Tensor input, const Tensor weight,
                  const std::optional<int64_t> padding_idx,
                  const std::optional<double> max_norm, const double norm_type,
                  const bool scale_grad_by_freq, const bool sparse,
                  Tensor out) const override {
    (void)padding_idx;
    (void)scale_grad_by_freq;
    (void)sparse;
    if (num_indices_ == 0 || embedding_dim_ == 0) {
      return;
    }
    if (max_norm.has_value()) {
      throw std::runtime_error("kunlun embedding: max_norm not supported");
    }

    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    const int tag = detail::KunlunDtypeTag(weight_dtype_);
    // Contiguity: indices (last dim stride 1) and output (last dim stride 1
    // with row-major layout). Non-contiguous rows fall back to an error —
    // InfiniLM always passes contiguous indices/outputs here.
    if ((input_strides_.size() >= 1 && input_strides_.back() != 1) ||
        out_strides_.back() != 1 ||
        (out_strides_.size() >= 2 &&
         out_strides_[out_strides_.size() - 2] !=
             static_cast<Tensor::Stride>(embedding_dim_))) {
      throw std::runtime_error(
          "kunlun embedding: unsupported index/output layout");
    }

    kunlun_launch::LaunchEmbedding(
        tag, input_dtype_ == DataType::kInt64 ? 1 : 0, out.data(), input.data(),
        weight.data(),
        (unsigned long long)num_indices_ * (unsigned long long)embedding_dim_,
        (int)embedding_dim_, (unsigned long long)vocab_size_, stream);
  }
};

}  // namespace infini::ops

#endif
