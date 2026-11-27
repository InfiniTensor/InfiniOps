#ifndef INFINI_OPS_METAX_CAST_KERNEL_CUH_
#define INFINI_OPS_METAX_CAST_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

namespace infini::ops::metax_cast_detail {

constexpr size_t kMaxRank = 16;

struct Layout {
  size_t rank = 0;

  size_t shape[kMaxRank]{};

  size_t strides[kMaxRank]{};
};

__device__ inline size_t Offset(size_t linear, const Layout& layout) {
  size_t offset = 0;
  for (size_t axis = layout.rank; axis > 0; --axis) {
    offset += (linear % layout.shape[axis - 1]) * layout.strides[axis - 1];
    linear /= layout.shape[axis - 1];
  }
  return offset;
}

template <bool kToFloat>
__global__ void CastKernel(const void* input, size_t count, Layout input_layout,
                           Layout output_layout, void* output) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(gridDim.x) * blockDim.x) {
    const size_t source = Offset(i, input_layout);
    const size_t destination = Offset(i, output_layout);
    if constexpr (kToFloat) {
      static_cast<float*>(output)[destination] = __uint_as_float(
          static_cast<uint32_t>(static_cast<const uint16_t*>(input)[source])
          << 16);
    } else {
      uint32_t bits = __float_as_uint(static_cast<const float*>(input)[source]);
      if ((bits & 0x7fffffffU) > 0x7f800000U) {
        bits = (bits >> 16) | 0x0040U;
      } else {
        bits = (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
      }
      static_cast<uint16_t*>(output)[destination] = static_cast<uint16_t>(bits);
    }
  }
}

}  // namespace infini::ops::metax_cast_detail

#endif
