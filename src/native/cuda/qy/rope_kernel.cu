#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "native/cuda/qy/data_type_.h"
#include "native/cuda/qy/rope_kernel.h"

namespace infini::ops::qy {
namespace {

template <typename T>
__device__ __forceinline__ float toFloat(T v);

template <>
__device__ __forceinline__ float toFloat<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ float toFloat<half>(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float toFloat<cuda_bfloat16>(cuda_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float v);

template <>
__device__ __forceinline__ float fromFloat<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ half fromFloat<half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ cuda_bfloat16 fromFloat<cuda_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// Flattened over (token, head, rotary_pair). Avoids seqlen×nhead 2D/3D grids.
template <typename T, bool IsGptJ>
__global__ void ropeFlatKernel(T *y, const T *x,
                               const int64_t *__restrict__ pos_ids,
                               const T *__restrict__ sin_table,
                               const T *__restrict__ cos_table, size_t seqlen,
                               size_t nhead, size_t head_dim, size_t table_dim,
                               size_t n_pairs) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  const size_t head_stride = head_dim;
  const size_t token_stride = nhead * head_dim;

  for (; idx < n_pairs; idx += stride) {
    size_t pair = idx % table_dim;
    size_t th = idx / table_dim;
    size_t head = th % nhead;
    size_t token = th / nhead;
    if (token >= seqlen) {
      continue;
    }

    const size_t vec_offset = token * token_stride + head * head_stride;
    const size_t pos_id = static_cast<size_t>(pos_ids[token]);
    const size_t table_offset = pos_id * table_dim + pair;
    const float sin_v = toFloat(sin_table[table_offset]);
    const float cos_v = toFloat(cos_table[table_offset]);

    size_t pos0;
    size_t pos1;
    if constexpr (IsGptJ) {
      pos0 = 2 * pair;
      pos1 = 2 * pair + 1;
    } else {
      pos0 = pair;
      pos1 = pair + table_dim;
    }

    const float x0 = toFloat(x[vec_offset + pos0]);
    const float x1 = toFloat(x[vec_offset + pos1]);
    y[vec_offset + pos0] = fromFloat<T>(x0 * cos_v - x1 * sin_v);
    y[vec_offset + pos1] = fromFloat<T>(x0 * sin_v + x1 * cos_v);
  }
}

template <typename T>
void launchRope(void *out, const void *in, const int64_t *pos_ids,
                const void *sin_table, const void *cos_table, size_t seqlen,
                size_t nhead, size_t head_dim, int algo_tag, void *stream) {
  if (seqlen == 0 || nhead == 0 || head_dim == 0) {
    return;
  }
  if ((head_dim & 1) != 0) {
    throw std::runtime_error("QY rope_launch: head_dim must be even");
  }

  const size_t table_dim = head_dim / 2;
  const size_t n_pairs = seqlen * nhead * table_dim;
  constexpr int block = 128;
  int grid = static_cast<int>((n_pairs + block - 1) / block);
  grid = std::min(grid, 1024);

  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  (void)cudaGetLastError();

  if (algo_tag == 0) {
    // GPT_J
    ropeFlatKernel<T, true><<<grid, block, 0, cuda_stream>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), pos_ids,
        reinterpret_cast<const T *>(sin_table),
        reinterpret_cast<const T *>(cos_table), seqlen, nhead, head_dim,
        table_dim, n_pairs);
  } else if (algo_tag == 1) {
    // GPT_NEOX
    ropeFlatKernel<T, false><<<grid, block, 0, cuda_stream>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), pos_ids,
        reinterpret_cast<const T *>(sin_table),
        reinterpret_cast<const T *>(cos_table), seqlen, nhead, head_dim,
        table_dim, n_pairs);
  } else {
    throw std::runtime_error("QY rope_launch: unsupported algo_tag " +
                             std::to_string(algo_tag));
  }

  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("QY rope_launch failed: ") +
                             cudaGetErrorString(err) +
                             " (seqlen=" + std::to_string(seqlen) +
                             ", nhead=" + std::to_string(nhead) +
                             ", head_dim=" + std::to_string(head_dim) +
                             ", grid=" + std::to_string(grid) +
                             ", block=" + std::to_string(block) + ")");
  }
}

}  // namespace

void rope_launch(void *out, const void *in, const int64_t *pos_ids,
                 const void *sin_table, const void *cos_table, size_t seqlen,
                 size_t nhead, size_t head_dim, int algo_tag, int dtype_tag,
                 void *stream) {
  using infini::rt::DataType;
  switch (static_cast<DataType>(dtype_tag)) {
    case DataType::kFloat16:
      launchRope<half>(out, in, pos_ids, sin_table, cos_table, seqlen, nhead,
                       head_dim, algo_tag, stream);
      return;
    case DataType::kBFloat16:
      launchRope<cuda_bfloat16>(out, in, pos_ids, sin_table, cos_table, seqlen,
                                nhead, head_dim, algo_tag, stream);
      return;
    case DataType::kFloat32:
      launchRope<float>(out, in, pos_ids, sin_table, cos_table, seqlen, nhead,
                        head_dim, algo_tag, stream);
      return;
    default:
      throw std::runtime_error("QY rope_launch: unsupported dtype_tag " +
                               std::to_string(dtype_tag));
  }
}

}  // namespace infini::ops::qy
