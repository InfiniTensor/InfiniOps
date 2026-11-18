#ifndef INFINI_OPS_KUNLUN_KERNEL_COMMONS_H_
#define INFINI_OPS_KUNLUN_KERNEL_COMMONS_H_

// Device-side common helpers for Kunlun XPU3 kernels. This header is only
// included by translation units compiled with the XTDK clang++ (`--xpu-arch
// =xpu3`).

#include <cstddef>
#include <cstdint>
#include <cstring>

// clang-format off
#include <xpu/kernel/xtdk.h>
#include <xpu/kernel/xtdk_atomic_sm_xpu3.h>
#include <xpu/kernel/xtdk_bf16.h>
#include <xpu/kernel/xtdk_fp16.h>
#include <xpu/kernel/xtdk_math.h>
#include <xpu/kernel/xtdk_simd.h>
#include <xpu/kernel/xtdk_trigonometric.h>
// clang-format on

namespace infini::ops {
namespace kunlun {

// XPU infers plain pointer kernel parameters into global address space 1;
// templating the pointer type lets the helper accept any address space.
template <typename SizePtr, typename StridePtr>
inline __device__ std::size_t IndexToOffset(std::size_t flat_index,
                                            std::size_t ndim, SizePtr shape,
                                            StridePtr strides) {
  std::size_t res = 0;
  for (std::size_t i = ndim; i-- > 0;) {
    res += (flat_index % shape[i]) * static_cast<std::size_t>(strides[i]);
    flat_index /= shape[i];
  }
  return res;
}

}  // namespace kunlun
}  // namespace infini::ops

#endif
