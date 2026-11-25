#ifndef INFINI_OPS_METAX_EMBEDDING_KERNEL_CUH_
#define INFINI_OPS_METAX_EMBEDDING_KERNEL_CUH_

#include <cstddef>
#include <cstdint>

namespace infini::ops {

template <typename T>
struct alignas(16) MetaxEmbeddingPack {
  T values[8];
};

template <typename T, typename IndexT>
__global__ void MetaxEmbeddingVectorizedKernel(T* out, const IndexT* indices,
                                               const T* weight,
                                               int64_t weight_row_stride,
                                               size_t vocab_size) {
  using Pack = MetaxEmbeddingPack<T>;
  const size_t token = blockIdx.x;
  const int64_t index = static_cast<int64_t>(indices[token]);

  // Match the generic kernel: invalid indices leave output rows untouched.
  if (index < 0 || static_cast<size_t>(index) >= vocab_size) {
    return;
  }

  auto output = reinterpret_cast<Pack*>(out + token * 4096);
  auto row = reinterpret_cast<const Pack*>(weight + index * weight_row_stride);
  for (int pack = threadIdx.x; pack < 512; pack += blockDim.x) {
    output[pack] = row[pack];
  }
}

}  // namespace infini::ops

#endif
