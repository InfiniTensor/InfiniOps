#ifndef INFINI_OPS_METAX_DEEPSEEK_MOE_KERNEL_H_
#define INFINI_OPS_METAX_DEEPSEEK_MOE_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// MACA's cu-bridge provides the CUDA-style half/bfloat16 headers;
// __nv_bfloat16 aliases __maca_bfloat16 (same include pattern as
// native/cuda/metax/ops/chunk_gated_delta_rule).
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "base/deepseek_moe.h"
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

inline size_t AlignUp(size_t value, size_t alignment = 256) {
  return (value + alignment - 1) / alignment * alignment;
}

inline void CheckMc(mcError_t status, const char* what) {
  if (status != mcSuccess) {
    throw std::runtime_error(std::string("MetaX DeepseekMoe ") + what +
                             " failed: " + std::to_string(int(status)));
  }
}

inline mcblasHandle_t GetHandle() {
  static thread_local mcblasHandle_t handle = []() {
    mcblasHandle_t h{};
    if (Blas<Device::Type::kMetax>::BlasCreate(&h) != MCBLAS_STATUS_SUCCESS)
      throw std::runtime_error(
          "MetaX DeepseekMoe: creating mcBLAS handle failed");
    return h;
  }();
  return handle;
}

struct WorkspaceLayout {
  size_t counts, offsets, positions, row_to_route, route_to_row, packed_hidden,
      gate_buf, up_buf, total;
};

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

__global__ void CountExpertsKernel(const int* topk_indices, int* counts,
                                   int routes, int num_experts) {
  const int route = blockIdx.x * blockDim.x + threadIdx.x;
  if (route >= routes) {
    return;
  }
  const int expert = topk_indices[route];
  if (expert >= 0 && expert < num_experts) {
    atomicAdd(counts + expert, 1);
  }
}

__global__ void PrefixCountsKernel(const int* counts, int* offsets,
                                   int num_experts) {
  extern __shared__ int scan[];
  const int tid = threadIdx.x;
  scan[tid] = tid < num_experts ? counts[tid] : 0;
  __syncthreads();
  for (int stride = 1; stride < blockDim.x; stride <<= 1) {
    int value = 0;
    if (tid >= stride) {
      value = scan[tid - stride];
    }
    __syncthreads();
    scan[tid] += value;
    __syncthreads();
  }
  if (tid == 0) {
    offsets[0] = 0;
  }
  if (tid < num_experts) {
    offsets[tid + 1] = scan[tid];
  }
}

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
               mcblasHandle_t handle) {
  using MetaxBlas = Blas<Device::Type::kMetax>;
  using MetaxRuntime = Runtime<Device::Type::kMetax>;
  const auto blas_dtype = std::is_same_v<T, __half> ? macaDataType(MACA_R_16F)
                                                    : macaDataType(MACA_R_16BF);
  const int hidden_i = static_cast<int>(hidden_size);
  const int inter_i = static_cast<int>(intermediate_size);
  const int topk_i = static_cast<int>(topk);
  const int experts_i = static_cast<int>(num_experts);

  auto* counts = reinterpret_cast<int*>(base + layout.counts);
  auto* offsets = reinterpret_cast<int*>(base + layout.offsets);
  auto* positions = reinterpret_cast<int*>(base + layout.positions);
  auto* row_to_route = reinterpret_cast<int*>(base + layout.row_to_route);
  auto* route_to_row = reinterpret_cast<int*>(base + layout.route_to_row);
  auto* packed = reinterpret_cast<T*>(base + layout.packed_hidden);
  auto* gate_buf = reinterpret_cast<T*>(base + layout.gate_buf);
  auto* up_buf = reinterpret_cast<T*>(base + layout.up_buf);

  std::vector<int> host_counts(num_experts + 1);
  std::vector<int> host_offsets(num_experts + 1);

  // Weights are row-major [out_features, in_features]; OP_T turns them
  // into column-major (in_features, out_features), matching the nvidia
  // run_gemm convention (m=out, n=tokens, k=in).
  auto gemm = [&](const void* a, const void* b, void* c, int m, int n, int k,
                  int lda, int ldb, int ldc) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const auto status = MetaxBlas::BlasGemmStridedBatchedEx(
        handle, MetaxBlas::BLAS_OP_T, MetaxBlas::BLAS_OP_N, m, n, k, &alpha, a,
        blas_dtype, lda, 0, b, blas_dtype, ldb, 0, &beta, c, blas_dtype, ldc, 0,
        1, MetaxBlas::BLAS_COMPUTE_32F, MetaxBlas::BLAS_GEMM_DEFAULT);
    if (status != MCBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("MetaX DeepseekMoe GEMM failed: " +
                               std::to_string(int(status)));
    }
  };

  for (size_t begin = 0; begin < ntokens; begin += kChunkTokens) {
    const int tokens =
        static_cast<int>(std::min(kChunkTokens, ntokens - begin));
    const int routes = tokens * topk_i;
    const T* hidden_chunk = hidden + begin * hidden_size;
    const int* ids_chunk = topk_indices + begin * topk_i;
    const float* weights_chunk = topk_weights + begin * topk_i;
    T* out_chunk = out + begin * hidden_size;

    CheckMc(MetaxRuntime::MemsetAsync(counts, 0,
                                      (num_experts + 1) * sizeof(int), stream),
            "memset");
    CheckMc(MetaxRuntime::MemsetAsync(positions, 0,
                                      (num_experts + 1) * sizeof(int), stream),
            "memset");
    CountExpertsKernel<<<(routes + kThreads - 1) / kThreads, kThreads, 0,
                         stream>>>(ids_chunk, counts, routes, experts_i);
    PrefixCountsKernel<<<1, kThreads, kThreads * sizeof(int), stream>>>(
        counts, offsets, experts_i);
    PackHiddenKernel<T><<<routes, kThreads, 0, stream>>>(
        hidden_chunk, ids_chunk, offsets, positions, row_to_route, route_to_row,
        packed, routes, hidden_i, topk_i, experts_i);
    CheckMc(MetaxRuntime::MemcpyAsync(
                host_counts.data(), counts, (num_experts + 1) * sizeof(int),
                MetaxRuntime::kMemcpyDeviceToHost, stream),
            "counts copy");
    CheckMc(MetaxRuntime::MemcpyAsync(
                host_offsets.data(), offsets, (num_experts + 1) * sizeof(int),
                MetaxRuntime::kMemcpyDeviceToHost, stream),
            "offsets copy");
    CheckMc(MetaxRuntime::StreamSynchronize(stream), "route sync");

    for (int e = 0; e < experts_i; ++e) {
      const int count = host_counts[e];
      if (count <= 0) {
        continue;
      }
      const size_t off = static_cast<size_t>(host_offsets[e]);
      gemm(gate_ptrs[e], packed + off * hidden_size,
           gate_buf + off * intermediate_size, inter_i, count, hidden_i,
           hidden_i, hidden_i, inter_i);
      gemm(up_ptrs[e], packed + off * hidden_size,
           up_buf + off * intermediate_size, inter_i, count, hidden_i, hidden_i,
           hidden_i, inter_i);
    }
    SwigluWeightKernel<T>
        <<<(static_cast<size_t>(routes) * intermediate_size + kThreads - 1) /
               kThreads,
           kThreads, 0, stream>>>(gate_buf, gate_buf, up_buf, weights_chunk,
                                  row_to_route, routes, inter_i);
    for (int e = 0; e < experts_i; ++e) {
      const int count = host_counts[e];
      if (count <= 0) {
        continue;
      }
      const size_t off = static_cast<size_t>(host_offsets[e]);
      gemm(down_ptrs[e], gate_buf + off * intermediate_size,
           packed + off * hidden_size, hidden_i, count, inter_i, inter_i,
           inter_i, hidden_i);
    }
    SumGroupedOutKernel<T>
        <<<(static_cast<size_t>(tokens) * hidden_size + kThreads - 1) /
               kThreads,
           kThreads, 0, stream>>>(out_chunk, packed, route_to_row, tokens,
                                  hidden_i, topk_i);
  }
  // Keep workspace reuse safe across chunks / subsequent layers.
  CheckMc(MetaxRuntime::StreamSynchronize(stream), "final sync");
}

}  // namespace metax_deepseek_moe_detail

// Native MetaX DeepseekMoe: expert-grouped GEMM path. The routing
// (count/scan/pack), activation and reduction kernels run on device and
// the per-expert projections go through mcBLAS, so no per-expert operator
// submits cross the dispatcher.
template <>
class Operator<DeepseekMoe, Device::Type::kMetax, 0> : public DeepseekMoe {
 public:
  using DeepseekMoe::DeepseekMoe;

  bool capture_safe() const override { return false; }

  size_t workspace_size_in_bytes() const override {
    const auto& hidden = metadata_[1];
    const size_t esz = hidden.element_size();
    const size_t ntokens = std::min<size_t>(
        hidden.size(0), metax_deepseek_moe_detail::kChunkTokens);
    const size_t routes = ntokens * metadata_[2].size(1);
    return metax_deepseek_moe_detail::Layout(num_experts_, routes,
                                             hidden.size(1), intermediate_, esz)
        .total;
  }

  void operator()(Tensor output, const Tensor hidden, const Tensor ids,
                  const Tensor weights, const std::vector<Tensor> gate,
                  const std::vector<Tensor> up, const std::vector<Tensor> down,
                  size_t intermediate, size_t num_experts) const override {
    namespace detail = metax_deepseek_moe_detail;
    using MetaxBlas = Blas<Device::Type::kMetax>;
    using MetaxRuntime = Runtime<Device::Type::kMetax>;

    Invocation(output, hidden, ids, weights, gate, up, down, intermediate,
               num_experts);
    if (hidden.size(0) == 0) {
      return;
    }
    moe_detail::Require(num_experts <= size_t(detail::kMaxExperts));
    moe_detail::Require(hidden.dtype() == DataType::kFloat16 ||
                        hidden.dtype() == DataType::kBFloat16);

    const size_t hidden_size = hidden.size(1);
    const size_t topk = ids.size(1);
    const size_t ntokens = hidden.size(0);
    const size_t esz = hidden.element_size();

    moe_detail::Workspace(
        workspace_, workspace_size_in_bytes_,
        detail::Layout(num_experts,
                       std::min(ntokens, detail::kChunkTokens) * topk,
                       hidden_size, intermediate, esz)
            .total,
        DeepseekMoe::Collect(output, hidden, ids, weights, gate, up, down));

    const auto stream =
        static_cast<MetaxRuntime::Stream>(stream_ ? stream_ : nullptr);
    auto handle = detail::GetHandle();
    MetaxBlas::BlasSetStream(handle, stream);

    std::vector<const void*> gate_ptrs(num_experts);
    std::vector<const void*> up_ptrs(num_experts);
    std::vector<const void*> down_ptrs(num_experts);
    for (size_t e = 0; e < num_experts; ++e) {
      gate_ptrs[e] = gate[e].data();
      up_ptrs[e] = up[e].data();
      down_ptrs[e] = down[e].data();
    }

    const auto layout = detail::Layout(
        num_experts, std::min(ntokens, detail::kChunkTokens) * topk,
        hidden_size, intermediate, esz);
    auto* base = static_cast<std::uint8_t*>(workspace_);

    if (hidden.dtype() == DataType::kFloat16) {
      detail::RunChunks<__half>(
          base, layout, ntokens, hidden_size, topk, intermediate, num_experts,
          reinterpret_cast<const __half*>(hidden.data()),
          reinterpret_cast<const int*>(ids.data()),
          reinterpret_cast<const float*>(weights.data()), gate_ptrs, up_ptrs,
          down_ptrs, reinterpret_cast<__half*>(output.data()), stream, handle);
    } else {
      detail::RunChunks<__nv_bfloat16>(
          base, layout, ntokens, hidden_size, topk, intermediate, num_experts,
          reinterpret_cast<const __nv_bfloat16*>(hidden.data()),
          reinterpret_cast<const int*>(ids.data()),
          reinterpret_cast<const float*>(weights.data()), gate_ptrs, up_ptrs,
          down_ptrs, reinterpret_cast<__nv_bfloat16*>(output.data()), stream,
          handle);
    }
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_DEEPSEEK_MOE_KERNEL_H_
