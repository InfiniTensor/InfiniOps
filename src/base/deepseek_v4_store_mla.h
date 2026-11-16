#ifndef INFINI_OPS_BASE_DEEPSEEK_V4_STORE_MLA_H_
#define INFINI_OPS_BASE_DEEPSEEK_V4_STORE_MLA_H_

#include "operator.h"

namespace infini::ops {

// Store BF16 or FP32 KV rows into DeepSeek V4's packed page format. Nonnegative
// slots must be unique within a launch; -1 skips a row.
class DeepseekV4StoreMla : public Operator<DeepseekV4StoreMla> {
 public:
  DeepseekV4StoreMla(const Tensor kv, const Tensor slots, int64_t page_tokens, Tensor pages) {
    (void)kv;
    (void)slots;
    (void)page_tokens;
    (void)pages;
  }

  virtual void operator()(const Tensor kv, const Tensor slots, int64_t page_tokens,
                          Tensor pages) const = 0;
};
}  // namespace infini::ops

#endif
