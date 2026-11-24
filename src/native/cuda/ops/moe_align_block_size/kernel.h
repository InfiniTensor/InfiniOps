#ifndef INFINI_OPS_CUDA_MOE_ALIGN_BLOCK_SIZE_KERNEL_H_
#define INFINI_OPS_CUDA_MOE_ALIGN_BLOCK_SIZE_KERNEL_H_

#include <optional>

#include "base/moe_align_block_size.h"
#include "device.h"
#include "native/cuda/ops/moe_align_block_size/kernel.cuh"

namespace infini::ops {
namespace moe_align_block_size_detail {

// Selects the tensor's device for the launch and restores the previous one.
template <typename Backend>
class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    if (Backend::GetDevice(&previous_) != Backend::kSuccess ||
        previous_ == device) {
      return;
    }
    restore_ = Backend::SetDevice(device) == Backend::kSuccess;
  }

  ~DeviceGuard() {
    if (restore_) {
      Backend::SetDevice(previous_);
    }
  }

 private:
  int previous_{-1};

  bool restore_{false};
};

}  // namespace moe_align_block_size_detail

// One block counts the routes per expert, writes the block plan and sorts the
// token indices into expert-aligned blocks.
template <typename Backend>
class CudaMoeAlignBlockSize : public MoeAlignBlockSize {
 public:
  using MoeAlignBlockSize::MoeAlignBlockSize;
  using MoeAlignBlockSize::operator();

  bool capture_safe() const override { return true; }

 protected:
  void Run(const Tensor topk_ids, const std::optional<Tensor> maybe_expert_map,
           const int64_t num_experts, const int64_t block_size,
           Tensor sorted_token_ids, Tensor experts_ids,
           Tensor num_tokens_post_pad) const override {
    moe_align_block_size_detail::DeviceGuard<Backend> guard{
        static_cast<int>(topk_ids_metadata_.device().index())};

    constexpr int32_t kThreads = 256;
    const auto shared_memory_size =
        static_cast<size_t>(num_experts_) * sizeof(int32_t);
    moe_align_block_size_detail::MoeAlignBlockSizeKernel<Backend::kDeviceType>
        <<<1, kThreads, shared_memory_size,
           static_cast<typename Backend::Stream>(stream_ ? stream_ : nullptr)>>>(
            reinterpret_cast<const int32_t*>(topk_ids.data()),
            maybe_expert_map
                ? reinterpret_cast<const int32_t*>(maybe_expert_map->data())
                : nullptr,
            reinterpret_cast<int32_t*>(sorted_token_ids.data()),
            reinterpret_cast<int32_t*>(experts_ids.data()),
            reinterpret_cast<int32_t*>(num_tokens_post_pad.data()), numel_,
            static_cast<int32_t>(num_experts_), static_cast<int32_t>(block_size_),
            sorted_token_ids_size_, experts_ids_size_,
            maybe_expert_map ? static_cast<int32_t>(maybe_expert_map->numel())
                             : 0);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_MOE_ALIGN_BLOCK_SIZE_KERNEL_H_
