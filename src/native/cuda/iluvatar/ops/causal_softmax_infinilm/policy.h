#ifndef INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_POLICY_H_
#define INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_POLICY_H_

#include <cstddef>

namespace infini::ops::iluvatar {

inline bool CanUseCausalSoftmaxFp32(bool bf16, bool same_device,
                                    bool contiguous, std::size_t heads,
                                    std::size_t query, std::size_t key) {
  // Bound the 2D launch before multiplying the row count.
  return bf16 && same_device && contiguous && heads > 0 && query > 1 &&
         query <= key && key <= 4096 &&
         heads <= (std::size_t{65535} * 65535) / query;
}

}  // namespace infini::ops::iluvatar

#endif
