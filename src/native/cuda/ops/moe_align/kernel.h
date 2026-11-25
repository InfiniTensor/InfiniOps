#ifndef INFINI_OPS_CUDA_MOE_ALIGN_KERNEL_H_
#define INFINI_OPS_CUDA_MOE_ALIGN_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "base/moe_align.h"
#include "native/cuda/ops/moe_align/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

namespace detail {

constexpr int kMoeAlignWarpSize = 32;

}  // namespace detail

template <typename Backend>
class CudaMoeAlign : public MoeAlign {
 public:
  CudaMoeAlign(Tensor sorted_token_ids, Tensor expert_ids,
               Tensor num_tokens_post_padded, Tensor topk_ids,
               Tensor expert_map, int64_t num_experts, int64_t block_size,
               bool pad_sorted_token_ids)
      : MoeAlign{sorted_token_ids, expert_ids,          num_tokens_post_padded,
                 topk_ids,         expert_map,          num_experts,
                 block_size,       pad_sorted_token_ids} {
    // Scratch buffer for the per-expert cumulative (exclusive) prefix offsets.
    const std::size_t cumsum_size =
        (static_cast<std::size_t>(num_experts) + 1) * sizeof(int32_t);
    Backend::Malloc(reinterpret_cast<void**>(&cumsum_buffer_), cumsum_size);
  }

  ~CudaMoeAlign() { Backend::Free(cumsum_buffer_); }

  void operator()(Tensor sorted_token_ids, Tensor expert_ids,
                  Tensor num_tokens_post_padded, Tensor topk_ids,
                  Tensor expert_map, int64_t num_experts, int64_t block_size,
                  bool pad_sorted_token_ids) const override {
    int threads = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();
    threads = ((threads + detail::kMoeAlignWarpSize - 1) /
               detail::kMoeAlignWarpSize) *
              detail::kMoeAlignWarpSize;

    const int32_t num_experts_shifted = static_cast<int32_t>(num_experts + 1);
    const int32_t block_size_i32 = static_cast<int32_t>(block_size);
    const int32_t max_num_tokens_padded =
        static_cast<int32_t>(max_num_tokens_padded_);
    const bool small_batch_expert_mode =
        (numel_ < 1024) && (num_experts_shifted <= 64);

    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    const int32_t* d_topk_ids =
        reinterpret_cast<const int32_t*>(topk_ids.data());
    const int32_t* d_expert_map =
        expert_map.numel() > 0
            ? reinterpret_cast<const int32_t*>(expert_map.data())
            : nullptr;
    int32_t* d_sorted_token_ids =
        reinterpret_cast<int32_t*>(sorted_token_ids.data());
    int32_t* d_expert_ids = reinterpret_cast<int32_t*>(expert_ids.data());
    int32_t* d_num_tokens_post_padded =
        reinterpret_cast<int32_t*>(num_tokens_post_padded.data());

    DispatchFunc<Backend::kDeviceType, DataType::kInt32>(
        topk_ids.dtype(),
        [&](auto type_tag) {
          using T = typename decltype(type_tag)::type;

          if (small_batch_expert_mode) {
            constexpr int32_t fill_threads = 256;
            const int32_t expert_threads =
                std::max(num_experts_shifted, detail::kMoeAlignWarpSize);
            const std::size_t shared_mem_size =
                (static_cast<std::size_t>(expert_threads + 1) *
                     static_cast<std::size_t>(num_experts_shifted) +
                 static_cast<std::size_t>(num_experts_shifted + 1)) *
                sizeof(int32_t);

            MoeAlignBlockSizeSmallBatchExpertKernel<T, fill_threads>
                <<<1, fill_threads + expert_threads, shared_mem_size,
                   cuda_stream>>>(d_topk_ids, d_expert_map, d_sorted_token_ids,
                                  d_expert_ids, d_num_tokens_post_padded,
                                  num_experts_shifted, block_size_i32, numel_,
                                  pad_sorted_token_ids, max_num_tokens_padded);
          } else {
            const int32_t scan_size = static_cast<int32_t>(
                detail::MoeAlignNextPow2(num_experts_shifted));
            const std::size_t shared_mem_size =
                (static_cast<std::size_t>(num_experts_shifted) +
                 static_cast<std::size_t>(num_experts_shifted + 1) +
                 static_cast<std::size_t>(scan_size) +
                 detail::kMoeAlignWarpSize) *
                sizeof(int32_t);

            MoeAlignBlockSizeKernel<T>
                <<<2, threads, shared_mem_size, cuda_stream>>>(
                    d_topk_ids, d_expert_map, d_sorted_token_ids, d_expert_ids,
                    d_num_tokens_post_padded, num_experts_shifted,
                    block_size_i32, numel_, cumsum_buffer_,
                    pad_sorted_token_ids, scan_size, max_num_tokens_padded);

            const int block_threads = std::min(256, threads);
            const int num_blocks =
                static_cast<int>((numel_ + block_threads - 1) / block_threads);
            const int max_blocks = 65535;
            const int actual_blocks = std::min(num_blocks, max_blocks);

            MoeAlignCountAndSortExpertTokensKernel<T>
                <<<actual_blocks, block_threads, 0, cuda_stream>>>(
                    d_topk_ids, d_expert_map, d_sorted_token_ids,
                    cumsum_buffer_, numel_);
          }
        },
        "CudaMoeAlign::operator()");
  }

 private:
  int32_t* cumsum_buffer_{nullptr};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_MOE_ALIGN_KERNEL_H_