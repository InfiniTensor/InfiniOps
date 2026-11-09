#ifndef INFINI_OPS_ILUVATAR_GEMM_GEMV_POLICY_H_
#define INFINI_OPS_ILUVATAR_GEMM_GEMV_POLICY_H_

#include <cstddef>
#include <cstdint>
#include <limits>

namespace infini::ops::iluvatar {

inline bool CanUseSmallBatchGemv(bool all_bf16, bool same_device, bool has_bias,
                                 float alpha, float beta, int trans_a,
                                 int trans_b, std::size_t m, std::size_t k,
                                 std::size_t n, std::ptrdiff_t a0,
                                 std::ptrdiff_t a1, std::ptrdiff_t b0,
                                 std::ptrdiff_t b1, std::ptrdiff_t y0,
                                 std::ptrdiff_t y1, std::uintptr_t a_address,
                                 std::uintptr_t b_address) {
  // Each lane loads two adjacent BF16 values. Reject offset views that are
  // not aligned for the packed load even when their strides match.
  constexpr std::size_t kMinWeightElements = 8 * 1024 * 1024;
  return all_bf16 && same_device && !has_bias && alpha == 1.0f &&
         beta == 0.0f && trans_a == 0 && trans_b == 0 && (m == 1 || m == 2) &&
         k >= 1024 && k % 64 == 0 && n >= 1024 && n % 4 == 0 &&
         k <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         n <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
         k >= (kMinWeightElements + n - 1) / n &&
         a0 == static_cast<std::ptrdiff_t>(k) && a1 == 1 && b0 == 1 &&
         b1 == static_cast<std::ptrdiff_t>(k) &&
         y0 == static_cast<std::ptrdiff_t>(n) && y1 == 1 &&
         a_address % 4 == 0 && b_address % 4 == 0;
}

}  // namespace infini::ops::iluvatar

#endif
