#ifndef INFINI_OPS_CUDA_MOE_FUSED_DENSE_KERNEL_H_
#define INFINI_OPS_CUDA_MOE_FUSED_DENSE_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "base/moe_fused_dense.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "native/cuda/ops/moe_fused_dense/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

// CUTLASS-compatible MoE fused-dense reference. Metax has no grouped GEMM, so
// the prefill and decode paths share one implementation: sorted tokens are
// packed into per-expert contiguous buckets, one GEMM runs per expert, SwiGLU +
// the down-projection run, and the weighted results are scattered back to the
// original tokens.
//
// The `Backend` type must expose the CUDA-like runtime members (`Malloc`,
// `Free`, `Memcpy`, `Memset`, `DeviceSynchronize`, `Stream`) and the BLAS
// members used here (`BlasHandle`, `BlasCreate`, `BlasSetStream`,
// `BlasGemmStridedBatchedEx`) plus `BLAS_OP_*` / dtype / algorithm constants.
template <typename Backend>
class CudaMoeFusedDense : public MoeFusedDense {
 public:
  CudaMoeFusedDense(Tensor output, Tensor hidden_states, Tensor w13, Tensor w2,
                    Tensor topk_weights, Tensor topk_ids,
                    Tensor sorted_token_ids, Tensor expert_ids,
                    Tensor num_tokens_post_padded)
      : MoeFusedDense{output,
                      hidden_states,
                      w13,
                      w2,
                      topk_weights,
                      topk_ids,
                      sorted_token_ids,
                      expert_ids,
                      num_tokens_post_padded} {}

  ~CudaMoeFusedDense() override = default;

  std::size_t workspace_size_in_bytes() const override {
    const std::size_t dtype_size = kDataTypeToSize.at(dtype_);
    const std::size_t pairs = num_tokens_ * topk_;
    std::size_t bytes = 0;
    bytes += AlignUp((num_experts_ + 1) * sizeof(int));
    bytes += AlignUp((num_experts_ + 1) * sizeof(int));
    bytes += AlignUp(pairs * sizeof(int));
    bytes += AlignUp(max_num_tokens_padded_ * hidden_size_ * dtype_size);
    bytes +=
        AlignUp(max_num_tokens_padded_ * intermediate_size_ * 2 * dtype_size);
    bytes += AlignUp(max_num_tokens_padded_ * intermediate_size_ * dtype_size);
    bytes += AlignUp(max_num_tokens_padded_ * hidden_size_ * dtype_size);
    return bytes;
  }

  void operator()(Tensor output, Tensor hidden_states, Tensor w13, Tensor w2,
                  Tensor topk_weights, Tensor topk_ids, Tensor sorted_token_ids,
                  Tensor expert_ids,
                  Tensor num_tokens_post_padded) const override {
    const int num_tokens = static_cast<int>(num_tokens_);
    const int hidden_size = static_cast<int>(hidden_size_);
    const int num_experts = static_cast<int>(num_experts_);
    const int intermediate_size = static_cast<int>(intermediate_size_);
    const int topk = static_cast<int>(topk_);
    const int num_tokens_padded = static_cast<int>(max_num_tokens_padded_);
    const int max_num_blocks = static_cast<int>(max_num_blocks_);
    const int pairs = num_tokens * topk;
    const int block_size = static_cast<int>(
        (max_num_tokens_padded_ + max_num_blocks_ - 1) / max_num_blocks_);
    const std::size_t dtype_size = kDataTypeToSize.at(dtype_);

    auto stream = static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    // ------------------------------------------------------------------
    // If the caller did not provide a workspace (Python binding currently
    // leaves Handle::workspace empty), allocate a temporary one ourselves.
    // ------------------------------------------------------------------
    const std::size_t ws_bytes = workspace_size_in_bytes();
    bool own_workspace = false;
    void* workspace_ptr = workspace_;
    if (workspace_ptr == nullptr) {
      Backend::Malloc(&workspace_ptr, ws_bytes);
      own_workspace = true;
    }

    // ------------------------------------------------------------------
    // Workspace layout: counts, offsets, permutation, then data buffers.
    // ------------------------------------------------------------------
    std::uint8_t* ptr = reinterpret_cast<std::uint8_t*>(workspace_ptr);
    int* counts = Advance<int>(ptr, num_experts + 1);
    int* offsets = Advance<int>(ptr, num_experts + 1);
    int* output_permutation = Advance<int>(ptr, pairs);
    void* packed_hidden =
        AdvanceBytes(ptr, static_cast<std::size_t>(num_tokens_padded) *
                              hidden_size * dtype_size);
    void* gate_up =
        AdvanceBytes(ptr, static_cast<std::size_t>(num_tokens_padded) *
                              intermediate_size * 2 * dtype_size);
    void* activated =
        AdvanceBytes(ptr, static_cast<std::size_t>(num_tokens_padded) *
                              intermediate_size * dtype_size);
    void* expert_out =
        AdvanceBytes(ptr, static_cast<std::size_t>(num_tokens_padded) *
                              hidden_size * dtype_size);

    auto& blas_handle = GetHandle();
    Backend::BlasSetStream(blas_handle, stream);

    Backend::Memset(output_permutation, 0xFF, pairs * sizeof(int));
    Backend::Memset(counts, 0, (num_experts + 1) * sizeof(int));
    Backend::Memset(
        expert_out, 0,
        static_cast<std::size_t>(num_tokens_padded) * hidden_size * dtype_size);

    // Read num_tokens_post_padded on the host before launch – Metax cannot
    // dereference a device pointer inside a kernel.
    int num_tokens_post_padded_host = 0;
    Backend::Memcpy(&num_tokens_post_padded_host, num_tokens_post_padded.data(),
                    sizeof(int), Backend::kMemcpyDeviceToHost);
    int num_blocks_aligned =
        (num_tokens_post_padded_host + block_size - 1) / block_size;

    CountAlignedExpertsKernel<<<(num_blocks_aligned + 255) / 256, 256, 0,
                                stream>>>(
        reinterpret_cast<const int*>(expert_ids.data()),
        num_tokens_post_padded_host, counts, num_experts, block_size);
    ExclusivePrefixCountsKernel<<<1, 1, 0, stream>>>(counts, offsets,
                                                     num_experts);

    DispatchFunc<Backend::kDeviceType, DataType::kFloat16, DataType::kBFloat16>(
        dtype_,
        [&](auto type_tag) {
          using T = typename decltype(type_tag)::type;
          PackHiddenAlignedKernel<Backend::kDeviceType, T>
              <<<num_tokens_padded, 256, 0, stream>>>(
                  reinterpret_cast<const T*>(hidden_states.data()),
                  reinterpret_cast<const int*>(sorted_token_ids.data()),
                  output_permutation, reinterpret_cast<T*>(packed_hidden),
                  pairs, topk, hidden_size, num_tokens_padded);
        },
        "CudaMoeFusedDense::PackHiddenAligned");

    // Per-expert GEMMs launch from the host, so copy counts/offsets back and
    // wait for the counting / packing kernels as well as the D2H copies.
    std::vector<int> host_counts(num_experts + 1);
    std::vector<int> host_offsets(num_experts + 1);
    Backend::Memcpy(host_counts.data(), counts, (num_experts + 1) * sizeof(int),
                    Backend::kMemcpyDeviceToHost);
    Backend::Memcpy(host_offsets.data(), offsets,
                    (num_experts + 1) * sizeof(int),
                    Backend::kMemcpyDeviceToHost);
    Backend::DeviceSynchronize();

    const auto blas_dtype =
        (dtype_ == DataType::kFloat16) ? Backend::R_16F : Backend::R_16BF;

    // GEMM1: gate_up (m x 2*I) = w13^T (2*I x H) @ packed_hidden^T.
    // w13 is col-major, dimensions (H, 2*I) -> OP_T with lda = H.
    {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      for (int e = 0; e < num_experts; ++e) {
        const int m = host_counts[e];
        if (m <= 0) {
          continue;
        }
        const int off = host_offsets[e];
        Backend::BlasGemmStridedBatchedEx(
            blas_handle, Backend::BLAS_OP_T, Backend::BLAS_OP_N,
            intermediate_size * 2, m, hidden_size, &alpha,
            reinterpret_cast<const std::uint8_t*>(w13.data()) +
                static_cast<std::size_t>(e) * intermediate_size * 2 *
                    hidden_size * dtype_size,
            blas_dtype, hidden_size, 0,
            reinterpret_cast<const std::uint8_t*>(packed_hidden) +
                static_cast<std::size_t>(off) * hidden_size * dtype_size,
            blas_dtype, hidden_size, 0, &beta,
            reinterpret_cast<std::uint8_t*>(gate_up) +
                static_cast<std::size_t>(off) * intermediate_size * 2 *
                    dtype_size,
            blas_dtype, intermediate_size * 2, 0, /*batch_count=*/1,
            Backend::BLAS_COMPUTE_32F, Backend::BLAS_GEMM_DEFAULT);
      }
    }

    DispatchFunc<Backend::kDeviceType, DataType::kFloat16, DataType::kBFloat16>(
        dtype_,
        [&](auto type_tag) {
          using T = typename decltype(type_tag)::type;
          const int total = num_tokens_padded * intermediate_size;
          SwigluKernel<Backend::kDeviceType, T>
              <<<(total + 255) / 256, 256, 0, stream>>>(
                  reinterpret_cast<const T*>(gate_up),
                  reinterpret_cast<T*>(activated), num_tokens_padded,
                  intermediate_size);
        },
        "CudaMoeFusedDense::Swiglu");

    // GEMM2: expert_out (m x H) = w2^T (H x I) @ activated^T.
    // w2 is col-major, dimensions (I, H) -> OP_T with lda = I.
    {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      for (int e = 0; e < num_experts; ++e) {
        const int m = host_counts[e];
        if (m <= 0) {
          continue;
        }
        const int off = host_offsets[e];
        Backend::BlasGemmStridedBatchedEx(
            blas_handle, Backend::BLAS_OP_T, Backend::BLAS_OP_N, hidden_size, m,
            intermediate_size, &alpha,
            reinterpret_cast<const std::uint8_t*>(w2.data()) +
                static_cast<std::size_t>(e) * hidden_size * intermediate_size *
                    dtype_size,
            blas_dtype, intermediate_size, 0,
            reinterpret_cast<const std::uint8_t*>(activated) +
                static_cast<std::size_t>(off) * intermediate_size * dtype_size,
            blas_dtype, intermediate_size, 0, &beta,
            reinterpret_cast<std::uint8_t*>(expert_out) +
                static_cast<std::size_t>(off) * hidden_size * dtype_size,
            blas_dtype, hidden_size, 0, /*batch_count=*/1,
            Backend::BLAS_COMPUTE_32F, Backend::BLAS_GEMM_DEFAULT);
      }
    }

    DispatchFunc<Backend::kDeviceType, DataType::kFloat16, DataType::kBFloat16>(
        dtype_,
        [&](auto type_tag) {
          using T = typename decltype(type_tag)::type;
          ApplyShuffleMulSumKernel<Backend::kDeviceType, T>
              <<<num_tokens, std::min(hidden_size, 1024), 0, stream>>>(
                  reinterpret_cast<const T*>(expert_out),
                  reinterpret_cast<T*>(output.data()), output_permutation,
                  reinterpret_cast<const float*>(topk_weights.data()),
                  num_tokens, topk, hidden_size);
        },
        "CudaMoeFusedDense::ApplyShuffleMulSum");
    if (own_workspace) {
      Backend::Free(workspace_ptr);
    }
  }

 protected:
  static std::size_t AlignUp(std::size_t value, std::size_t alignment = 16) {
    return (value + alignment - 1) / alignment * alignment;
  }

  template <typename T>
  static T* Advance(std::uint8_t*& ptr, std::size_t count) {
    T* out = reinterpret_cast<T*>(ptr);
    ptr += AlignUp(count * sizeof(T));
    return out;
  }

  static void* AdvanceBytes(std::uint8_t*& ptr, std::size_t bytes) {
    void* out = reinterpret_cast<void*>(ptr);
    ptr += AlignUp(bytes);
    return out;
  }

  static typename Backend::BlasHandle& GetHandle() {
    thread_local typename Backend::BlasHandle handle = []() {
      typename Backend::BlasHandle h;
      Backend::BlasCreate(&h);
      return h;
    }();
    return handle;
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_MOE_FUSED_DENSE_KERNEL_H_