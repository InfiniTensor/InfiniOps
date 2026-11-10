#ifndef INFINI_OPS_CUDA_MOE_FUSED_DENSE_KERNEL_H_
#define INFINI_OPS_CUDA_MOE_FUSED_DENSE_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "base/moe_fused_dense.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/moe_fused_dense/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

// MoE fused-dense reference for CUDA-like backends without a
// grouped GEMM (e.g. Metax): sorted tokens are packed into
// per-expert contiguous buckets, one GEMM runs per expert, SwiGLU
// + the down-projection run, and the weighted results are
// scattered back to the original tokens.
//
// The `Backend` type must expose the CUDA-like runtime members
// (`Stream`, `Malloc`, `Free`, `Memcpy`, `Memset`,
// `DeviceSynchronize`, `kMemcpyDeviceToHost`) and the BLAS
// members used here (`BlasHandle`, `BlasCreate`, `BlasSetStream`,
// `BlasGemmStridedBatchedEx`, `BLAS_OP_T`, `BLAS_OP_N`, `R_16F`,
// `R_16BF`, `BLAS_COMPUTE_32F`, `BLAS_GEMM_DEFAULT`).
template <typename Backend>
class CudaMoeFusedDense : public MoeFusedDense {
 public:
  CudaMoeFusedDense(Tensor output, const Tensor hidden_states, const Tensor w13,
                    const Tensor w2, const Tensor topk_weights,
                    const Tensor topk_ids, const Tensor sorted_token_ids,
                    const Tensor expert_ids,
                    const Tensor num_tokens_post_padded, size_t block_size,
                    const std::optional<Tensor> bias1 = std::nullopt,
                    const std::optional<Tensor> bias2 = std::nullopt)
      : MoeFusedDense{output,
                      hidden_states,
                      w13,
                      w2,
                      topk_weights,
                      topk_ids,
                      sorted_token_ids,
                      expert_ids,
                      num_tokens_post_padded,
                      block_size,
                      bias1,
                      bias2} {
    moe_detail::Require(
        hidden_states.IsContiguous() && w13.IsContiguous() &&
        w2.IsContiguous() && topk_weights.IsContiguous() &&
        topk_ids.IsContiguous() && sorted_token_ids.IsContiguous() &&
        expert_ids.IsContiguous() && num_tokens_post_padded.IsContiguous() &&
        output.IsContiguous());
    moe_detail::Require(hidden_states.dtype() == DataType::kFloat16 ||
                        hidden_states.dtype() == DataType::kBFloat16);
    if (bias1) {
      moe_detail::Require(bias1->IsContiguous());
    }
    if (bias2) {
      moe_detail::Require(bias2->IsContiguous());
    }
  }

  ~CudaMoeFusedDense() override = default;

  std::size_t workspace_size_in_bytes() const override {
    const auto& metadata_ = MoeFusedDense::metadata_;
    const auto dtype = metadata_[1].dtype();
    const std::size_t dtype_size = kDataTypeToSize.at(dtype);
    const std::size_t num_tokens = metadata_[1].size(0);
    const std::size_t hidden_size = metadata_[1].size(1);
    const std::size_t intermediate_size = metadata_[3].size(2);
    const std::size_t topk = metadata_[5].size(1);
    const std::size_t max_num_tokens_padded = metadata_[6].numel();
    const std::size_t pairs = num_tokens * topk;
    std::size_t bytes = 0;
    bytes += AlignUp((num_experts() + 1) * sizeof(int));
    bytes += AlignUp((num_experts() + 1) * sizeof(int));
    bytes += AlignUp(pairs * sizeof(int));
    bytes += AlignUp(max_num_tokens_padded * hidden_size * dtype_size);
    bytes +=
        AlignUp(max_num_tokens_padded * intermediate_size * 2 * dtype_size);
    bytes += AlignUp(max_num_tokens_padded * intermediate_size * dtype_size);
    bytes += AlignUp(max_num_tokens_padded * hidden_size * dtype_size);
    return bytes;
  }

  bool capture_safe() const override { return false; }

  void operator()(Tensor output, const Tensor hidden_states, const Tensor w13,
                  const Tensor w2, const Tensor weights, const Tensor ids,
                  const Tensor sorted_token_ids, const Tensor expert_ids,
                  const Tensor num_tokens_post_padded, size_t block_size,
                  const std::optional<Tensor> bias1,
                  const std::optional<Tensor> bias2) const override {
    Invocation(output, hidden_states, w13, w2, weights, ids, sorted_token_ids,
               expert_ids, num_tokens_post_padded, block_size, bias1, bias2);

    const int num_tokens = static_cast<int>(hidden_states.size(0));
    const int hidden_size = static_cast<int>(hidden_states.size(1));
    const int num_experts = static_cast<int>(w13.size(0));
    const int intermediate_size = static_cast<int>(w2.size(2));
    const int topk = static_cast<int>(ids.size(1));
    const int num_tokens_padded = static_cast<int>(sorted_token_ids.numel());
    const int max_num_tokens_padded =
        static_cast<int>(sorted_token_ids.numel());
    const int pairs = num_tokens * topk;
    const int block_size_i = static_cast<int>(block_size);
    const std::size_t dtype_size = kDataTypeToSize.at(hidden_states.dtype());

    auto stream = static_cast<typename Backend::Stream>(stream_);

    void* workspace_ptr = workspace_;
    moe_detail::Require(workspace_ptr != nullptr &&
                        workspace_size_in_bytes_ >= workspace_size_in_bytes());

    // Workspace layout: counts, offsets, permutation, then data buffers.
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

    const bool has_bias1 = bias1.has_value();
    const bool has_bias2 = bias2.has_value();

    Backend::Memset(output_permutation, 0xFF, pairs * sizeof(int));
    Backend::Memset(counts, 0, (num_experts + 1) * sizeof(int));
    Backend::Memset(
        expert_out, 0,
        static_cast<std::size_t>(num_tokens_padded) * hidden_size * dtype_size);

    // Read num_tokens_post_padded on the host before launching the
    // counting / packing kernels.
    int num_tokens_post_padded_host = 0;
    Backend::Memcpy(&num_tokens_post_padded_host, num_tokens_post_padded.data(),
                    sizeof(int), Backend::kMemcpyDeviceToHost);
    int num_blocks_aligned =
        (num_tokens_post_padded_host + block_size_i - 1) / block_size_i;

    moe_fused_dense_detail::CountAlignedExpertsKernel<<<
        (num_blocks_aligned + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const int*>(expert_ids.data()),
        num_tokens_post_padded_host, counts, num_experts, block_size_i);
    moe_fused_dense_detail::ExclusivePrefixCountsKernel<Backend::kDeviceType>
        <<<1, 1, 0, stream>>>(counts, offsets, num_experts);

    DispatchFunc<List<DataType::kFloat16, DataType::kBFloat16>>(
        {static_cast<int64_t>(hidden_states.dtype())},
        [&](auto type_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(type_tag)>;
          moe_fused_dense_detail::PackHiddenAlignedKernel<Backend::kDeviceType,
                                                          T>
              <<<num_tokens_padded, 256, 0, stream>>>(
                  reinterpret_cast<const T*>(hidden_states.data()),
                  reinterpret_cast<const int*>(sorted_token_ids.data()),
                  output_permutation, reinterpret_cast<T*>(packed_hidden),
                  pairs, topk, hidden_size, num_tokens_padded);
        },
        "CudaMoeFusedDense::PackHiddenAligned");

    // Per-expert GEMMs launch from the host, so copy counts/offsets
    // back and wait for the counting / packing kernels as well as the
    // D2H copies.
    std::vector<int> host_counts(num_experts + 1);
    std::vector<int> host_offsets(num_experts + 1);
    Backend::Memcpy(host_counts.data(), counts, (num_experts + 1) * sizeof(int),
                    Backend::kMemcpyDeviceToHost);
    Backend::Memcpy(host_offsets.data(), offsets,
                    (num_experts + 1) * sizeof(int),
                    Backend::kMemcpyDeviceToHost);
    Backend::DeviceSynchronize();

    const auto blas_dtype = (hidden_states.dtype() == DataType::kFloat16)
                                ? Backend::R_16F
                                : Backend::R_16BF;

    // Bias rows are written into the packed buffers before each GEMM
    // so rows the GEMM never touches still receive the bias.
    if (has_bias1 || has_bias2) {
      DispatchFunc<List<DataType::kFloat16, DataType::kBFloat16>>(
          {static_cast<int64_t>(hidden_states.dtype())},
          [&](auto type_tag) {
            using T = TypeMapType<Backend::kDeviceType, ListGet<0>(type_tag)>;
            moe_fused_dense_detail::FillPrefillBiasKernel<
                Backend::kDeviceType, T><<<num_experts, 256, 0, stream>>>(
                has_bias1 ? reinterpret_cast<T*>(gate_up) : nullptr,
                has_bias2 ? reinterpret_cast<T*>(expert_out) : nullptr,
                has_bias1 ? reinterpret_cast<const T*>(bias1->data()) : nullptr,
                has_bias2 ? reinterpret_cast<const T*>(bias2->data()) : nullptr,
                host_counts.data(), host_offsets.data(), intermediate_size,
                hidden_size);
          },
          "CudaMoeFusedDense::FillBias");
    }

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

    DispatchFunc<List<DataType::kFloat16, DataType::kBFloat16>>(
        {static_cast<int64_t>(hidden_states.dtype())},
        [&](auto type_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(type_tag)>;
          moe_fused_dense_detail::SwigluKernel<Backend::kDeviceType, T>
              <<<num_tokens_padded * intermediate_size / 256 + 1, 256, 0,
                 stream>>>(reinterpret_cast<const T*>(gate_up),
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

    DispatchFunc<List<DataType::kFloat16, DataType::kBFloat16>>(
        {static_cast<int64_t>(hidden_states.dtype())},
        [&](auto type_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(type_tag)>;
          moe_fused_dense_detail::ApplyShuffleMulSumKernel<Backend::kDeviceType,
                                                           T>
              <<<num_tokens, std::min(hidden_size, 1024), 0, stream>>>(
                  reinterpret_cast<const T*>(expert_out),
                  reinterpret_cast<T*>(output.data()), output_permutation,
                  reinterpret_cast<const float*>(weights.data()), num_tokens,
                  topk, hidden_size);
        },
        "CudaMoeFusedDense::ApplyShuffleMulSum");
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

 private:
  std::size_t num_experts() const {
    return MoeFusedDense::metadata_[2].size(0);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_MOE_FUSED_DENSE_KERNEL_H_
