#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_SPARSE_MLA_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_SPARSE_MLA_H_

#include "operator.h"

namespace infini::ops {

// Reference FlashMLA ABI: queries [rows, heads, 512]; KV pages are packed
// FP8 NoPE plus BF16 RoPE and separate per-token exponent bytes.
class DeepseekV4SparseMla : public Operator<DeepseekV4SparseMla> {
 public:
  DeepseekV4SparseMla(const Tensor query, const Tensor swa, const Tensor indices,
                      const Tensor lengths, const Tensor sink, const std::optional<Tensor> extra,
                      const std::optional<Tensor> extra_indices, const std::optional<Tensor> extra_lengths,
                      int64_t extra_page_size, double scale, Tensor out) {
    (void)query;
    (void)swa;
    (void)indices;
    (void)lengths;
    (void)sink;
    (void)extra;
    (void)extra_indices;
    (void)extra_lengths;
    (void)extra_page_size;
    (void)scale;
    (void)out;
  }

  virtual void operator()(const Tensor query, const Tensor swa, const Tensor indices,
                          const Tensor lengths, const Tensor sink,
                          const std::optional<Tensor> extra,
                          const std::optional<Tensor> extra_indices,
                          const std::optional<Tensor> extra_lengths,
                          int64_t extra_page_size, double scale, Tensor out) const = 0;
};

}  // namespace infini::ops

#endif
