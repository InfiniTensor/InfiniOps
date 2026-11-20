#ifndef INFINI_OPS_METAX_FUSED_GATED_DELTA_NET_GATING_KERNEL_H_
#define INFINI_OPS_METAX_FUSED_GATED_DELTA_NET_GATING_KERNEL_H_

#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>

#include "base/fused_gated_delta_net_gating.h"
#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
// cuda_fp16.h / cuda_bf16.h come from the MACA cu-bridge,
// where __nv_bfloat16 aliases __maca_bfloat16.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace infini::ops {
namespace metax_gating_detail {
struct Layout {
  ptrdiff_t strides[6][3]{};
  size_t total, seq, hidden;
  float beta, threshold;
};
template <typename T>
__global__ void kernel(const T* A, const T* a, const T* b, const T* dt,
                       float* g, float* betao, Layout p) {
  const size_t linear =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= p.total) return;
  const size_t h = linear % p.hidden, s = (linear / p.hidden) % p.seq;
  const size_t batch = linear / p.hidden / p.seq;
  ptrdiff_t offsets[6];
  for (int i = 0; i < 6; ++i)
    offsets[i] = static_cast<ptrdiff_t>(batch) * p.strides[i][0] +
                 static_cast<ptrdiff_t>(s) * p.strides[i][1] +
                 static_cast<ptrdiff_t>(h) * p.strides[i][2];
  const float x = static_cast<float>(a[offsets[1]]) +
                  static_cast<float>(dt[h * p.strides[3][0]]);
  const float bx = p.beta * x;
  const float softplus = bx <= p.threshold ? log1pf(expf(bx)) / p.beta : x;
  g[offsets[4]] = -expf(static_cast<float>(A[h * p.strides[0][0]])) * softplus;
  const float bv = static_cast<float>(b[offsets[2]]);
  const float z = expf(bv >= 0 ? -bv : bv);
  betao[offsets[5]] = bv >= 0 ? 1.0f / (1.0f + z) : z / (1.0f + z);
}
}  // namespace metax_gating_detail

template <>
class Operator<FusedGatedDeltaNetGating, Device::Type::kMetax>
    : public FusedGatedDeltaNetGating {
 public:
  using FusedGatedDeltaNetGating::FusedGatedDeltaNetGating;
  using FusedGatedDeltaNetGating::operator();

  bool capture_safe() const override { return true; }

  void operator()(Tensor A, Tensor a, Tensor b, Tensor dt, float beta,
                  float threshold, Tensor g, Tensor betao) const override {
    auto& p = const_cast<metax_gating_detail::Layout&>(p_);
    std::array<Tensor, 6> tensors{A, a, b, dt, g, betao};
    for (size_t i = 0; i < 6; ++i)
      if (!std::equal_to<Tensor>{}(tensors[i], metadata_[i]))
        throw std::invalid_argument("gating metadata changed after Make");
    if (std::memcmp(&beta, &beta_, sizeof(float)) ||
        std::memcmp(&threshold, &threshold_, sizeof(float)))
      throw std::invalid_argument("gating attributes changed after Make");
    if (total_ / 256 + (total_ % 256 != 0) >
        std::numeric_limits<unsigned int>::max())
      throw std::overflow_error("gating grid exceeds CUDA limit");
    p.total = total_;
    p.seq = seq_;
    p.hidden = hidden_;
    p.beta = beta_;
    p.threshold = threshold_;
    for (size_t i = 0; i < 6; ++i)
      for (size_t j = 0; j < metadata_[i].strides().size(); ++j)
        p.strides[i][j] = metadata_[i].strides()[j];
    if (!total_) return;
    const dim3 grid(
        static_cast<unsigned int>(total_ / 256 + (total_ % 256 != 0)));
    const auto stream =
        static_cast<typename Runtime<Device::Type::kMetax>::Stream>(
            stream_ ? stream_ : nullptr);
    switch (metadata_[1].dtype()) {
      case DataType::kFloat32:
        launch<float>(A, a, b, dt, g, betao, grid, stream);
        break;
      case DataType::kFloat16:
        launch<half>(A, a, b, dt, g, betao, grid, stream);
        break;
      case DataType::kBFloat16:
        launch<__nv_bfloat16>(A, a, b, dt, g, betao, grid, stream);
        break;
      default:
        throw std::invalid_argument("gating dtype unsupported");
    }
  }

 private:
  mutable metax_gating_detail::Layout p_{};
  template <typename T>
  void launch(Tensor A, Tensor a, Tensor b, Tensor dt, Tensor g, Tensor betao,
              dim3 grid,
              typename Runtime<Device::Type::kMetax>::Stream stream) const {
    metax_gating_detail::kernel<T><<<grid, 256, 0, stream>>>(
        static_cast<const T*>(A.data()), static_cast<const T*>(a.data()),
        static_cast<const T*>(b.data()), static_cast<const T*>(dt.data()),
        static_cast<float*>(g.data()), static_cast<float*>(betao.data()), p_);
  }
};

}  // namespace infini::ops

#endif
