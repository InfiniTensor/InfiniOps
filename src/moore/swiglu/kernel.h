#ifndef INFINI_OPS_MOORE_SWIGLU_KERNEL_H_
#define INFINI_OPS_MOORE_SWIGLU_KERNEL_H_

#include <utility>

// clang-format off
#include <musa_runtime.h>
// clang-format on

#include "cuda/swiglu/kernel.h"

namespace infini::ops {

namespace swiglu {

template <typename T>
struct MooreSwigluOp {
  __device__ __forceinline__ T operator()(const T& up, const T& gate) const {
    if constexpr (std::is_same_v<T, half>) {
      float gatef = __half2float(gate);
      half sig =
          __float2half(__frcp_rn(__fadd_rn(1.0f, __expf(-gatef))));
      return __hmul(__hmul(gate, sig), up);
    } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
      float gate0 = __low2float(gate);
      float gate1 = __high2float(gate);
      float sig0 = __frcp_rn(__fadd_rn(1.0f, __expf(-gate0)));
      float sig1 = __frcp_rn(__fadd_rn(1.0f, __expf(-gate1)));
      float up0 = __low2float(up);
      float up1 = __high2float(up);
      float res0 = __fmul_rn(__fmul_rn(gate0, sig0), up0);
      float res1 = __fmul_rn(__fmul_rn(gate1, sig1), up1);
      return __floats2bfloat162_rn(res0, res1);
    } else {
      return CudaSwigluOp<T>{}(up, gate);
    }
  }
};

struct MooreBackend {
  using stream_t = musaStream_t;

  static constexpr auto malloc = [](auto&&... args) {
    return musaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto memcpy = [](auto&&... args) {
    return musaMemcpy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto free = [](auto&&... args) {
    return musaFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto memcpyH2D = musaMemcpyHostToDevice;
};

}  // namespace swiglu

template <>
class Operator<Swiglu, Device::Type::kMoore>
    : public CudaSwiglu<swiglu::MooreBackend, swiglu::MooreSwigluOp> {
 public:
  using CudaSwiglu<swiglu::MooreBackend, swiglu::MooreSwigluOp>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
