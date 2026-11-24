#ifndef INFINI_OPS_CUDA_FUSED_GATED_DELTA_NET_GATING_KERNEL_H_
#define INFINI_OPS_CUDA_FUSED_GATED_DELTA_NET_GATING_KERNEL_H_

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "base/fused_gated_delta_net_gating.h"
#include "device.h"
#include "native/cuda/ops/fused_gated_delta_net_gating/kernel.cuh"

namespace infini::ops {

// Gates of a gated delta net: the log decay from `A`, `a` and `dt_bias`, and
// the delta-rule coefficient as the sigmoid of `b`.
template <typename Backend>
class CudaFusedGatedDeltaNetGating : public FusedGatedDeltaNetGating {
 public:
  CudaFusedGatedDeltaNetGating(Tensor A, Tensor a, Tensor b, Tensor dt,
                               float beta, float threshold, Tensor g,
                               Tensor betao)
      : FusedGatedDeltaNetGating(A, a, b, dt, beta, threshold, g, betao) {
    if (device_.type() != Backend::kDeviceType)
      throw std::invalid_argument("gating requires its provider's device");
    check_device();
    if (total_ / 256 + (total_ % 256 != 0) >
        std::numeric_limits<unsigned int>::max())
      throw std::overflow_error("gating grid exceeds the launch limit");
    p_.total = total_;
    p_.seq = seq_;
    p_.hidden = hidden_;
    p_.beta = beta;
    p_.threshold = threshold;
    for (size_t i = 0; i < 6; ++i)
      for (size_t j = 0; j < metadata_[i].strides().size(); ++j)
        p_.strides[i][j] = metadata_[i].strides()[j];
  }

  // All metadata is a kernel value argument. There is no allocation, host copy,
  // synchronization or data-dependent host work in operator(); stream launch
  // is capturable. Make owns no device resources and requires zero workspace.
  bool capture_safe() const override { return true; }

  void operator()(Tensor A, Tensor a, Tensor b, Tensor dt, float beta,
                  float threshold, Tensor g, Tensor betao) const override {
    check_device();
    std::array<Tensor, 6> tensors{A, a, b, dt, g, betao};
    for (size_t i = 0; i < 6; ++i)
      if (!std::equal_to<Tensor>{}(tensors[i], metadata_[i]))
        throw std::invalid_argument("gating metadata changed after Make");
    if (std::memcmp(&beta, &beta_, sizeof(float)) ||
        std::memcmp(&threshold, &threshold_, sizeof(float)))
      throw std::invalid_argument("gating attributes changed after Make");
    if (!total_) return;
    const dim3 grid(
        static_cast<unsigned int>(total_ / 256 + (total_ % 256 != 0)));
    const auto stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : nullptr);
    switch (metadata_[1].dtype()) {
      case DataType::kFloat32:
        launch<float>(A, a, b, dt, g, betao, grid, stream);
        break;
      case DataType::kFloat16:
        launch<TypeMapType<Backend::kDeviceType, DataType::kFloat16>>(
            A, a, b, dt, g, betao, grid, stream);
        break;
      case DataType::kBFloat16:
        launch<TypeMapType<Backend::kDeviceType, DataType::kBFloat16>>(
            A, a, b, dt, g, betao, grid, stream);
        break;
      default:
        throw std::invalid_argument("gating dtype unsupported");
    }
  }

 private:
  gating_detail::Layout p_{};

  void check_device() const {
    int current = -1;
    if (Backend::GetDevice(&current) != Backend::kSuccess)
      throw std::runtime_error("gating device query failed");
    if (current != device_.index())
      throw std::invalid_argument("gating owner device mismatch");
  }

  template <typename T>
  void launch(Tensor A, Tensor a, Tensor b, Tensor dt, Tensor g, Tensor betao,
              dim3 grid, typename Backend::Stream stream) const {
    gating_detail::GatingKernel<Backend::kDeviceType, T><<<grid, 256, 0, stream>>>(
        static_cast<const T*>(A.data()), static_cast<const T*>(a.data()),
        static_cast<const T*>(b.data()), static_cast<const T*>(dt.data()),
        static_cast<float*>(g.data()), static_cast<float*>(betao.data()), p_);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_FUSED_GATED_DELTA_NET_GATING_KERNEL_H_
