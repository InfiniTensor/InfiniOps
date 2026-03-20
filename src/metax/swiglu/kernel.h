#ifndef INFINI_OPS_METAX_SWIGLU_KERNEL_H_
#define INFINI_OPS_METAX_SWIGLU_KERNEL_H_

#include <utility>

// clang-format off
#include <mcr/mc_runtime.h>
// clang-format on

#include "cuda/swiglu/kernel.h"

namespace infini::ops {

namespace swiglu {

struct MetaxBackend {
  using stream_t = mcStream_t;

  static constexpr auto malloc = [](auto&&... args) {
    return mcMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto memcpy = mcMemcpy;

  static constexpr auto free = mcFree;

  static constexpr auto memcpyH2D = mcMemcpyHostToDevice;
};

}  // namespace swiglu

template <>
class Operator<Swiglu, Device::Type::kMetax>
    : public CudaSwiglu<swiglu::MetaxBackend> {
 public:
  using CudaSwiglu<swiglu::MetaxBackend>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
