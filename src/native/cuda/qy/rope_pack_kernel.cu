#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "native/cuda/qy/rope_pack_kernel.h"

namespace infini::ops::qy {
namespace {

// One thread per (token, head) vector; copy contiguous last-dim of `dim` elems.
template <typename T>
__global__ void packVecKernel(T *dst, const T *src, size_t n_vecs, size_t dim,
                              size_t heads, int64_t s0, int64_t s1) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; idx < n_vecs; idx += stride) {
    size_t head = idx % heads;
    size_t token = idx / heads;
    const T *row = src + token * s0 + head * s1;
    T *out = dst + idx * dim;
    // Vectorize 8-byte chunks when dim and alignment allow (bf16/fp16 dim=128).
    size_t d = 0;
    if (dim >= 4 && (reinterpret_cast<uintptr_t>(row) % 8) == 0 &&
        (reinterpret_cast<uintptr_t>(out) % 8) == 0) {
      for (; d + 4 <= dim; d += 4) {
        *reinterpret_cast<uint64_t *>(out + d) =
            *reinterpret_cast<const uint64_t *>(row + d);
      }
    }
    for (; d < dim; ++d) {
      out[d] = row[d];
    }
  }
}

template <typename T>
__global__ void unpackVecKernel(T *dst, const T *src, size_t n_vecs, size_t dim,
                                size_t heads, int64_t s0, int64_t s1) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; idx < n_vecs; idx += stride) {
    size_t head = idx % heads;
    size_t token = idx / heads;
    T *row = dst + token * s0 + head * s1;
    const T *in = src + idx * dim;
    size_t d = 0;
    if (dim >= 4 && (reinterpret_cast<uintptr_t>(row) % 8) == 0 &&
        (reinterpret_cast<uintptr_t>(in) % 8) == 0) {
      for (; d + 4 <= dim; d += 4) {
        *reinterpret_cast<uint64_t *>(row + d) =
            *reinterpret_cast<const uint64_t *>(in + d);
      }
    }
    for (; d < dim; ++d) {
      row[d] = in[d];
    }
  }
}

template <typename T>
void launchTyped(bool pack, void *dst, const void *src, size_t seq,
                 size_t heads, size_t dim, int64_t stride0, int64_t stride1,
                 int64_t stride2, void *stream) {
  if (stride2 != 1) {
    throw std::runtime_error(
        "QY rope_pack/unpack: only contiguous last-dim (stride2==1) supported");
  }
  const size_t n_vecs = seq * heads;
  if (n_vecs == 0) {
    return;
  }
  constexpr int block = 128;
  int grid = static_cast<int>((n_vecs + block - 1) / block);
  // Prefer covering all vectors in one wave when Denglin allows.
  grid = std::min(grid, 4096);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  (void)cudaGetLastError();
  if (pack) {
    packVecKernel<T><<<grid, block, 0, cuda_stream>>>(
        reinterpret_cast<T *>(dst), reinterpret_cast<const T *>(src), n_vecs,
        dim, heads, stride0, stride1);
  } else {
    unpackVecKernel<T><<<grid, block, 0, cuda_stream>>>(
        reinterpret_cast<T *>(dst), reinterpret_cast<const T *>(src), n_vecs,
        dim, heads, stride0, stride1);
  }
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    // Fall back to smaller grid if Denglin rejects large launch.
    grid = std::min(grid, 128);
    if (pack) {
      packVecKernel<T><<<grid, block, 0, cuda_stream>>>(
          reinterpret_cast<T *>(dst), reinterpret_cast<const T *>(src), n_vecs,
          dim, heads, stride0, stride1);
    } else {
      unpackVecKernel<T><<<grid, block, 0, cuda_stream>>>(
          reinterpret_cast<T *>(dst), reinterpret_cast<const T *>(src), n_vecs,
          dim, heads, stride0, stride1);
    }
    err = cudaGetLastError();
  }
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string(pack ? "QY rope_pack_launch failed: "
                         : "QY rope_unpack_launch failed: ") +
        cudaGetErrorString(err) + " (n_vecs=" + std::to_string(n_vecs) +
        ", dim=" + std::to_string(dim) + ")");
  }
}

void launchPackOrUnpack(bool pack, void *dst, const void *src, size_t seq,
                        size_t heads, size_t dim, int64_t stride0,
                        int64_t stride1, int64_t stride2, int elem_bytes,
                        void *stream) {
  if (seq == 0 || heads == 0 || dim == 0) {
    return;
  }
  if (elem_bytes == 2) {
    launchTyped<uint16_t>(pack, dst, src, seq, heads, dim, stride0, stride1,
                          stride2, stream);
  } else if (elem_bytes == 4) {
    launchTyped<uint32_t>(pack, dst, src, seq, heads, dim, stride0, stride1,
                          stride2, stream);
  } else {
    throw std::runtime_error("QY rope_pack: elem_bytes must be 2 or 4");
  }
}

}  // namespace

void rope_pack_launch(void *dst, const void *src, size_t seq, size_t heads,
                      size_t dim, int64_t src_stride0, int64_t src_stride1,
                      int64_t src_stride2, int elem_bytes, void *stream) {
  launchPackOrUnpack(true, dst, src, seq, heads, dim, src_stride0, src_stride1,
                     src_stride2, elem_bytes, stream);
}

void rope_unpack_launch(void *dst, const void *src, size_t seq, size_t heads,
                        size_t dim, int64_t dst_stride0, int64_t dst_stride1,
                        int64_t dst_stride2, int elem_bytes, void *stream) {
  launchPackOrUnpack(false, dst, src, seq, heads, dim, dst_stride0, dst_stride1,
                     dst_stride2, elem_bytes, stream);
}

}  // namespace infini::ops::qy
