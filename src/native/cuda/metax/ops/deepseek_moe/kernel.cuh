#ifndef INFINI_OPS_METAX_DEEPSEEK_MOE_KERNEL_CUH_
#define INFINI_OPS_METAX_DEEPSEEK_MOE_KERNEL_CUH_

// Shared declarations of the MetaX DeepseekMoe provider: constants, the
// workspace layout, the workspace lease, and the device-side template
// kernels. The registration lives in kernel.h; kernel.maca is the single
// compilation unit that defines the host scheduling (including the
// non-template routing kernels) and explicitly instantiates RunChunks
// for the two supported element types.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// MACA's cu-bridge provides the CUDA-style half/bfloat16 headers;
// __nv_bfloat16 aliases __maca_bfloat16 (same include pattern as
// native/cuda/metax/ops/chunk_gated_delta_rule).
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "native/cuda/metax/blas.h"
#include "native/cuda/metax/runtime_.h"

namespace infini::ops {
namespace metax_deepseek_moe_detail {

constexpr int kThreads = 256;
// The prefix-scan kernel runs in a single block of kThreads threads.
// 021 routes over 80 experts; nvidia's grouped path also caps at 256.
constexpr int kMaxExperts = 256;
// Tokens processed per pass so the intermediate workspace stays bounded
// for long-context prefill (021 supports 256k tokens).
constexpr size_t kChunkTokens = 4096;

struct WorkspaceLayout {
  size_t counts, offsets, positions, row_to_route, route_to_row, packed_hidden,
      gate_buf, up_buf, total;
};

inline size_t AlignUp(size_t value, size_t alignment = 256) {
  return (value + alignment - 1) / alignment * alignment;
}

inline WorkspaceLayout Layout(size_t num_experts, size_t routes,
                              size_t hidden_size, size_t intermediate_size,
                              size_t element_size) {
  WorkspaceLayout l{};
  l.counts = 0;
  // Counts and offsets stay adjacent so one D2H copy fetches both.
  l.offsets = l.counts + (num_experts + 1) * sizeof(int);
  l.positions = AlignUp(l.offsets + (num_experts + 1) * sizeof(int));
  l.row_to_route = AlignUp(l.positions + (num_experts + 1) * sizeof(int));
  l.route_to_row = l.row_to_route + routes * sizeof(int);
  l.packed_hidden = AlignUp(l.route_to_row + routes * sizeof(int));
  l.gate_buf = AlignUp(l.packed_hidden + routes * hidden_size * element_size);
  l.up_buf = AlignUp(l.gate_buf + routes * intermediate_size * element_size);
  l.total = AlignUp(l.up_buf + routes * intermediate_size * element_size);
  return l;
}

// Stream-ordered workspace lease for one operator invocation. Callers with a
// dispatch-provided workspace use it directly (validated); callers without one
// (python bindings) fall back to an eager MallocAsync that is released in
// stream order, so it outlives every kernel of the submission.
class WorkspaceLease {
 public:
  WorkspaceLease(void* external, std::size_t available, std::size_t required,
                 Runtime<Device::Type::kMetax>::Stream stream);
  WorkspaceLease(const WorkspaceLease&) = delete;
  WorkspaceLease& operator=(const WorkspaceLease&) = delete;
  ~WorkspaceLease();
  void* data() const { return data_; }

 private:
  void* data_{nullptr};
  Runtime<Device::Type::kMetax>::Stream stream_;
  bool owned_{false};
};

// Pack every routed (token, expert) pair into expert-contiguous rows and
// record both index directions of the permutation.
template <typename T>
__global__ void PackHiddenKernel(const T* hidden, const int* topk_indices,
                                 const int* offsets, int* positions,
                                 int* row_to_route, int* route_to_row,
                                 T* packed_hidden, int routes, int hidden_size,
                                 int topk, int num_experts) {
  const int route = blockIdx.x;
  if (route >= routes) {
    return;
  }
  const int expert = topk_indices[route];
  if (expert < 0 || expert >= num_experts) {
    return;
  }
  const int tid = threadIdx.x;
  __shared__ int row;
  if (tid == 0) {
    const int local = atomicAdd(positions + expert, 1);
    row = offsets[expert] + local;
    row_to_route[row] = route;
    route_to_row[route] = row;
  }
  __syncthreads();
  const int token = route / topk;
  for (int h = tid; h < hidden_size; h += blockDim.x) {
    packed_hidden[static_cast<size_t>(row) * hidden_size + h] =
        hidden[static_cast<size_t>(token) * hidden_size + h];
  }
}

// In-place SiLU(gate) * up * topk_weight over the expert-packed buffer.
template <typename T>
__global__ void SwigluWeightKernel(T* activated, const T* gate, const T* up,
                                   const float* topk_weights,
                                   const int* row_to_route, int rows,
                                   int intermediate_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * intermediate_size;
  if (idx >= total) {
    return;
  }
  const int row = idx / intermediate_size;
  const int route = row_to_route[row];
  const float g = static_cast<float>(gate[idx]);
  const float u = static_cast<float>(up[idx]);
  const float silu = g / (1.0f + expf(-g));
  activated[idx] = static_cast<T>(silu * u * topk_weights[route]);
}

// Sum the topk expert contributions back per (token, hidden) element.
template <typename T>
__global__ void SumGroupedOutKernel(T* out, const T* packed_out,
                                    const int* route_to_row, int ntokens,
                                    int hidden_size, int topk) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = ntokens * hidden_size;
  if (idx >= total) {
    return;
  }
  const int token = idx / hidden_size;
  const int h = idx - token * hidden_size;
  float acc = 0.0f;
  for (int k = 0; k < topk; ++k) {
    const int route = token * topk + k;
    const int row = route_to_row[route];
    acc += static_cast<float>(
        packed_out[static_cast<size_t>(row) * hidden_size + h]);
  }
  out[idx] = static_cast<T>(acc);
}

// Expert-grouped MoE: pack routed tokens by expert, run gate/up/down GEMMs
// per expert (batch=1 strided GEMM, the pattern moe_fused_dense already
// exercises on MetaX), apply SiLU*up*weight in place, then sum the topk
// contributions per token. Large prefills are chunked so the workspace
// stays bounded.
template <typename T>
void RunChunks(std::uint8_t* base, const WorkspaceLayout& layout,
               size_t ntokens, size_t hidden_size, size_t topk,
               size_t intermediate_size, size_t num_experts, const T* hidden,
               const int* topk_indices, const float* topk_weights,
               const std::vector<const void*>& gate_ptrs,
               const std::vector<const void*>& up_ptrs,
               const std::vector<const void*>& down_ptrs, T* out,
               Runtime<Device::Type::kMetax>::Stream stream,
               mcblasHandle_t handle);

// mcBLAS handle is created lazily once per thread; defined in kernel.maca.
mcblasHandle_t GetHandle();

// The two element types the provider dispatches on; defined in kernel.maca.
extern template void RunChunks<__half>(std::uint8_t*, const WorkspaceLayout&,
                                       size_t, size_t, size_t, size_t, size_t,
                                       const __half*, const int*, const float*,
                                       const std::vector<const void*>&,
                                       const std::vector<const void*>&,
                                       const std::vector<const void*>&, __half*,
                                       Runtime<Device::Type::kMetax>::Stream,
                                       mcblasHandle_t);
extern template void RunChunks<__nv_bfloat16>(
    std::uint8_t*, const WorkspaceLayout&, size_t, size_t, size_t, size_t,
    size_t, const __nv_bfloat16*, const int*, const float*,
    const std::vector<const void*>&, const std::vector<const void*>&,
    const std::vector<const void*>&, __nv_bfloat16*,
    Runtime<Device::Type::kMetax>::Stream, mcblasHandle_t);

}  // namespace metax_deepseek_moe_detail
}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_DEEPSEEK_MOE_KERNEL_CUH_
