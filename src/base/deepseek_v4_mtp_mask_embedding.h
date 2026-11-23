#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_MTP_MASK_EMBEDDING_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_MTP_MASK_EMBEDDING_H_

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class DeepseekV4MtpMaskEmbedding
    : public Operator<DeepseekV4MtpMaskEmbedding> {
 public:
  static constexpr bool kAutomaticDefault = true;

  DeepseekV4MtpMaskEmbedding(Tensor embedding, const Tensor positions) {}

  virtual void operator()(Tensor embedding,
                          const Tensor positions) const = 0;
};

}  // namespace infini::ops

#endif
