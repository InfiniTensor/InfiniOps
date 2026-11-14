#ifndef INFINI_OPS_CUDA_LIGHTNING_ATTENTION_INFINILM_KERNEL_CUH_
#define INFINI_OPS_CUDA_LIGHTNING_ATTENTION_INFINILM_KERNEL_CUH_

#include <cstddef>

#include "native/cuda/caster.cuh"
#include "native/cuda/kernel_commons.cuh"

namespace infini::ops {

/// One block per `(batch, head)` and one thread per state/output column.
///
/// The recurrent state `[head_dim, head_dim]` of the destination pool row is
/// staged first (the source row is copied into it when the two rows differ, so
/// the source row stays untouched), then updated in place:
///
///   state[i][j] = ratio * state[i][j] + k[i] * v[j]
///   out[j]      = sum_i q[i] * state[i][j]
///
/// Thread `j` owns column `j`, so the state update needs no cross-thread
/// synchronization; only the shared `k`/`q` rows do. The kernel must be
/// launched with exactly `head_dim` threads so that every thread reaches the
/// barriers.
template <Device::Type kDev, typename Data, typename Index>
__global__ void LightningAttentionInfinilmKernel(
    Data* out, Data* state_pool, const Data* q, const Data* k, const Data* v,
    const float* slope, const Index* initial_state_indices,
    const Index* final_state_indices, size_t seq_len, size_t head_dim,
    ptrdiff_t state_pool_stride, ptrdiff_t state_head_stride,
    ptrdiff_t state_row_stride, ptrdiff_t q_batch_stride, ptrdiff_t q_seq_stride,
    ptrdiff_t q_head_stride, ptrdiff_t k_batch_stride, ptrdiff_t k_seq_stride,
    ptrdiff_t k_head_stride, ptrdiff_t v_batch_stride, ptrdiff_t v_seq_stride,
    ptrdiff_t v_head_stride, ptrdiff_t out_batch_stride, ptrdiff_t out_seq_stride,
    ptrdiff_t out_head_stride, ptrdiff_t slope_stride) {
  const size_t batch = blockIdx.y;
  const size_t head = blockIdx.x;
  const size_t column = threadIdx.x;

  const size_t initial_row = static_cast<size_t>(initial_state_indices[batch]);
  const size_t final_row = static_cast<size_t>(final_state_indices[batch]);

  Data* state = state_pool + final_row * state_pool_stride +
                head * state_head_stride;
  if (initial_row != final_row) {
    const Data* source = state_pool + initial_row * state_pool_stride +
                         head * state_head_stride;
    for (size_t index = column; index < head_dim * head_dim;
         index += blockDim.x) {
      const size_t i = index / head_dim;
      const size_t j = index % head_dim;
      state[i * state_row_stride + j] = source[i * state_row_stride + j];
    }
  }
  __syncthreads();

  extern __shared__ float shared[];
  float* shared_k = shared;
  float* shared_q = shared + head_dim;

  const float ratio = expf(-slope[head * slope_stride]);
  const Data* q_head = q + batch * q_batch_stride + head * q_head_stride;
  const Data* k_head = k + batch * k_batch_stride + head * k_head_stride;
  const Data* v_head = v + batch * v_batch_stride + head * v_head_stride;
  Data* out_head = out + batch * out_batch_stride + head * out_head_stride;

  for (size_t t = 0; t < seq_len; ++t) {
    shared_k[column] = Caster<kDev>::template Cast<float>(
        k_head[t * k_seq_stride + column]);
    shared_q[column] = Caster<kDev>::template Cast<float>(
        q_head[t * q_seq_stride + column]);
    __syncthreads();

    const float v_column = Caster<kDev>::template Cast<float>(
        v_head[t * v_seq_stride + column]);
    for (size_t i = 0; i < head_dim; ++i) {
      Data* element = state + i * state_row_stride + column;
      const float updated = ratio * Caster<kDev>::template Cast<float>(*element) +
                            shared_k[i] * v_column;
      *element = Caster<kDev>::template Cast<Data>(updated);
    }

    float accumulator = 0.0f;
    for (size_t i = 0; i < head_dim; ++i) {
      accumulator += shared_q[i] * Caster<kDev>::template Cast<float>(
                                       state[i * state_row_stride + column]);
    }
    out_head[t * out_seq_stride + column] =
        Caster<kDev>::template Cast<Data>(accumulator);

    // The next iteration overwrites the shared rows, so all threads must have
    // finished reading them.
    __syncthreads();
  }
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_LIGHTNING_ATTENTION_INFINILM_KERNEL_CUH_
