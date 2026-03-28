#ifndef INFINI_OPS_COMMON_CUDA_KERNEL_COMMONS_H_
#define INFINI_OPS_COMMON_CUDA_KERNEL_COMMONS_H_

#include "caster.h"

namespace infini::ops {

using AllCudaBlockSizes = List<128, 256, 512, 1024, 2048>;

__forceinline__ __device__ __host__ size_t
IndexToOffset(size_t flat_index, size_t ndim, const size_t* shape,
              const ptrdiff_t* strides) {
  size_t res = 0;
  for (size_t i = ndim; i-- > 0;) {
    res += (flat_index % shape[i]) * strides[i];
    flat_index /= shape[i];
  }
  return res;
}

}  // namespace infini::ops

#endif
