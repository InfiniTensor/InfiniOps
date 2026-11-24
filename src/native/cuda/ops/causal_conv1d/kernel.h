#ifndef INFINI_OPS_CUDA_CAUSAL_CONV1D_KERNEL_H_
#define INFINI_OPS_CUDA_CAUSAL_CONV1D_KERNEL_H_

#include <climits>
#include <optional>
#include <stdexcept>

#include "base/causal_conv1d.h"
#include "device.h"
#include "native/cuda/ops/causal_conv1d/kernel.cuh"

namespace infini::ops {

// Depthwise causal K=4 convolution over batched or packed request sequences.
// The last three inputs of every request are committed to its state slot.
template <typename Backend>
class CudaCausalConv1d : public CausalConv1d {
 public:
  CudaCausalConv1d(const Tensor qkv, const Tensor weight, Tensor conv_state,
                   const std::optional<Tensor> bias,
                   const std::optional<Tensor> cu_seqlens,
                   const std::optional<Tensor> initial_state_indices,
                   const std::optional<Tensor> final_state_indices, Tensor out,
                   std::optional<Tensor> final_conv_state)
      : CausalConv1d(qkv, weight, conv_state, bias, cu_seqlens,
                     initial_state_indices, final_state_indices, out,
                     final_conv_state) {
    if (device_.type() != Backend::kDeviceType)
      throw std::invalid_argument("causal conv requires its provider's device");
    if (weight.shape()[2] != 4 || conv_state.shape()[2] != 3)
      throw std::invalid_argument(
          "causal conv kernel requires K4 and state_len3");
    if (tokens_ > static_cast<size_t>(std::numeric_limits<int64_t>::max() - 3))
      throw std::overflow_error(
          "causal conv history index exceeds int64");
    check_device();
    // Baseline uses int request/channel variables and CUDA's grid.y dimension.
    if (requests_ >= static_cast<size_t>(INT_MAX) || channels_ > 65535)
      throw std::invalid_argument(
          "causal conv request/channel launch extent unsupported");
    p_.tokens = tokens_;
    p_.channels = channels_;
    p_.pool = pool_;
    for (size_t i = 0; i < metadata_.size(); ++i)
      if (metadata_[i])
        for (size_t j = 0; j < metadata_[i]->strides().size(); ++j)
          p_.strides[i][j] = metadata_[i]->strides()[j];
    p_.cu_i64 = metadata_[4] && metadata_[4]->dtype() == DataType::kInt64;
    p_.initial_i64 = metadata_[5] && metadata_[5]->dtype() == DataType::kInt64;
    p_.final_i64 = metadata_[6] && metadata_[6]->dtype() == DataType::kInt64;
  }

  // Device capture/state-pool lifetime is not supported for this stateful path.
  // Zero device allocations does not alone establish capture eligibility.
  void operator()(const Tensor qkv, const Tensor weight, Tensor conv_state,
                  const std::optional<Tensor> bias,
                  const std::optional<Tensor> cu_seqlens,
                  const std::optional<Tensor> initial_state_indices,
                  const std::optional<Tensor> final_state_indices, Tensor out,
                  std::optional<Tensor> final_conv_state) const override {
    const Views views{qkv,
                      weight,
                      conv_state,
                      bias,
                      cu_seqlens,
                      initial_state_indices,
                      final_state_indices,
                      out,
                      final_conv_state};
    for (size_t i = 0; i < views.size(); ++i) {
      const auto& t = views[i];
      if (t.has_value() != metadata_[i].has_value() ||
          (t && (!std::equal_to<Tensor>{}(*t, *metadata_[i]) ||
                 (t->numel() && !t->data()))))
        throw std::invalid_argument(
            "causal conv metadata changed or nonempty storage is null");
    }
    check_device();
    if (!requests_ || !channels_) return;
    switch (dtype_) {
      case DataType::kFloat32:
        launch<float>(views);
        break;
      case DataType::kFloat16:
        launch<TypeMapType<Backend::kDeviceType, DataType::kFloat16>>(views);
        break;
      case DataType::kBFloat16:
        launch<TypeMapType<Backend::kDeviceType, DataType::kBFloat16>>(views);
        break;
      default:
        throw std::invalid_argument("causal conv dtype unsupported");
    }
  }

 private:
  causal_conv1d_detail::Layout p_{};

  void check_device() const {
    int current = -1;
    if (Backend::GetDevice(&current) != Backend::kSuccess)
      throw std::runtime_error("causal conv device query failed");
    if (current != device_.index())
      throw std::invalid_argument("causal conv owner device mismatch");
  }

  template <typename T>
  void launch(const Views& v) const {
    auto pointer = [&](size_t i) -> void* {
      return v[i] ? const_cast<void*>(v[i]->data()) : nullptr;
    };
    const dim3 grid(static_cast<unsigned int>(requests_),
                    static_cast<unsigned int>(channels_), 1);
    const auto stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : nullptr);
    causal_conv1d_detail::CausalConv1dKernel<Backend::kDeviceType, T>
        <<<grid, 1, 0, stream>>>(
            static_cast<const T*>(pointer(0)), static_cast<const T*>(pointer(1)),
            static_cast<T*>(pointer(2)), static_cast<const T*>(pointer(3)),
            pointer(4), pointer(5), pointer(6), static_cast<T*>(pointer(7)),
            static_cast<T*>(pointer(8)), p_);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CAUSAL_CONV1D_KERNEL_H_
