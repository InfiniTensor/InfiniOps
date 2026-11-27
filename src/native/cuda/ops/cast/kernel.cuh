#ifndef INFINI_OPS_CUDA_CAST_KERNEL_CUH_
#define INFINI_OPS_CUDA_CAST_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

#include "data_type.h"

namespace infini::ops {
namespace cast_detail {

// Shape and strides of a strided operand pair, in device memory.
struct Layout {
  const std::size_t* shape;
  const std::ptrdiff_t* input_strides;
  const std::ptrdiff_t* output_strides;
  std::size_t rank{};
  bool contiguous{};
};

template <Device::Type kDevice, typename InT, typename OutT>
__global__ void CastKernel(const InT* input, OutT* out, std::size_t first,
                           std::size_t count, Layout layout) {
  const auto local =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local >= count) return;
  std::size_t input_offset = first + local, output_offset = input_offset;
  if (!layout.contiguous) {
    std::size_t remaining = input_offset;
    input_offset = output_offset = 0;
    for (std::size_t axis = layout.rank; axis > 0; --axis) {
      const auto coordinate = remaining % layout.shape[axis - 1];
      remaining /= layout.shape[axis - 1];
      input_offset += coordinate * layout.input_strides[axis - 1];
      output_offset += coordinate * layout.output_strides[axis - 1];
    }
  }
  out[output_offset] =
      Caster<kDevice>::template Cast<OutT>(input[input_offset]);
}

}  // namespace cast_detail
}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CAST_KERNEL_CUH_
