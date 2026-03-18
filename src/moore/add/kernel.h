#ifndef INFINI_OPS_MOORE_ADD_KERNEL_H_
#define INFINI_OPS_MOORE_ADD_KERNEL_H_

#include <utility>

// clang-format off
#include <musa_runtime.h>
// clang-format on

#include "cuda/add/kernel.h"

namespace infini::ops {

namespace add {

template <typename T>
struct MooreAddOp {
  __device__ __forceinline__ T operator()(const T& input,
                                          const T& other) const {
    if constexpr (std::is_same_v<T, TypeMapType<DataType::kBFloat16>>) {
      return input + other;
    } else {
      return CudaAddOp<T>{}(input, other);
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

}  // namespace add

template <>
class Operator<Add, Device::Type::kMoore>
    : public CudaAdd<add::MooreBackend, add::MooreAddOp> {
 public:
  using CudaAdd<add::MooreBackend, add::MooreAddOp>::CudaAdd;
};

}  // namespace infini::ops

#endif
