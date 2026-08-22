#ifndef INFINI_OPS_NVIDIA_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_H_
#define INFINI_OPS_NVIDIA_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_H_

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "base/top_k_top_p_sampling_from_logits.h"

namespace infini::ops {

template <>
class Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia, 0>
    : public TopKTopPSamplingFromLogits {
 public:
  Operator(const Tensor logits, const Tensor top_k, const Tensor top_p,
           const std::optional<Tensor> indices,
           const std::string filter_apply_order, const bool deterministic,
           const bool check_nan, const std::optional<int64_t> seed,
           const std::optional<int64_t> offset, Tensor out);

  ~Operator() override;

  std::size_t workspace_size_in_bytes() const override;

  void operator()(const Tensor logits, const Tensor top_k, const Tensor top_p,
                  const std::optional<Tensor> indices,
                  const std::string filter_apply_order,
                  const bool deterministic, const bool check_nan,
                  const std::optional<int64_t> seed,
                  const std::optional<int64_t> offset,
                  Tensor out) const override;

 private:
  static void ValidateSupportedOptions(const std::string& filter_apply_order,
                                       bool deterministic, bool check_nan);

  static void ValidateHostTensor(const Tensor tensor);

  static void ValidateIndices(const std::optional<Tensor>& indices);

  static int64_t ReadTopK(const Tensor top_k, Tensor::Size row);

  static double ReadTopP(const Tensor top_p, Tensor::Size row);

  static int64_t ReadIndex(const Tensor indices, Tensor::Size row);

  static std::size_t DispatchWorkspaceSize(DataType dtype,
                                           Tensor::Size vocab_size);

  struct DefaultWorkspaceSlot {
    void* workspace{nullptr};
    cudaStream_t stream{nullptr};
    cudaEvent_t completion{nullptr};
    bool completion_recorded{false};
  };

  DefaultWorkspaceSlot* AcquireDefaultWorkspaceSlot(cudaStream_t stream) const;

  static void RecordDefaultWorkspaceUse(DefaultWorkspaceSlot* slot,
                                        cudaStream_t stream);

  int device_index_{0};

  std::size_t workspace_size_{0};

  static constexpr std::size_t kDefaultWorkspaceSlotCount = 2;

  mutable std::array<DefaultWorkspaceSlot, kDefaultWorkspaceSlotCount>
      default_workspace_slots_{};

  mutable std::size_t next_default_workspace_slot_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_H_
