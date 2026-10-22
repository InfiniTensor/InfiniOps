#ifndef INFINI_OPS_CUDA_PAD_KERNEL_CUH_
#define INFINI_OPS_CUDA_PAD_KERNEL_CUH_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/caster.cuh"

namespace infini::ops::cuda_pad_detail {

struct Layout {
  std::int64_t ndim = 0;
  std::int64_t input_shape[16]{};
  std::int64_t output_shape[16]{};
  std::int64_t input_strides[16]{};
  std::int64_t output_strides[16]{};
  std::int64_t left_padding[16]{};
};

template <typename T>
__global__ void PadKernel(const T* input, T* out, Layout layout, T fill,
                          std::size_t count) {
  const auto start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t linear = start; linear < count; linear += step) {
    std::size_t remaining = linear;
    std::int64_t input_offset = 0, output_offset = 0;
    bool inside = true;
    for (std::int64_t d = layout.ndim; d-- > 0;) {
      const auto coord = static_cast<std::int64_t>(
          remaining % static_cast<std::size_t>(layout.output_shape[d]));
      remaining /= static_cast<std::size_t>(layout.output_shape[d]);
      output_offset += coord * layout.output_strides[d];
      const auto source = coord - layout.left_padding[d];
      if (source < 0 || source >= layout.input_shape[d])
        inside = false;
      else
        input_offset += source * layout.input_strides[d];
    }
    out[output_offset] = inside ? input[input_offset] : fill;
  }
}

template <typename Backend>
void Launch(const void* input, void* out, DataType dtype, double value,
            Layout layout, std::size_t count, typename Backend::Stream stream) {
  if (!count) return;
  constexpr unsigned kThreads = 256;
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>((count - 1) / kThreads + 1, 65535));
  DispatchFunc<Backend::kDeviceType, AllFloatTypes>(
      dtype,
      [&](auto tag) {
        using T = typename decltype(tag)::type;
        // Preserve the original float intermediate for reduced-precision fill.
        const T fill = [&] {
          if constexpr (IsFP16<Backend::kDeviceType, T> ||
                        IsBFloat16<Backend::kDeviceType, T>)
            return Caster<Backend::kDeviceType>::template Cast<T>(
                static_cast<float>(value));
          else
            return static_cast<T>(value);
        }();
        PadKernel<T><<<blocks, kThreads, 0, stream>>>(
            static_cast<const T*>(input), static_cast<T*>(out), layout, fill,
            count);
      },
      "CudaPad::Launch");
  const auto status = Backend::PeekAtLastError();
  if (status != Backend::kSuccess)
    throw std::runtime_error(Backend::GetErrorString(status));
}

}  // namespace infini::ops::cuda_pad_detail

#endif  // INFINI_OPS_CUDA_PAD_KERNEL_CUH_
