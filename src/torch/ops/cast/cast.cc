#include "torch/ops/cast/cast.h"

#include <cstdint>
#include <numeric>

#include "torch/c10.h"
#ifdef WITH_NVIDIA
#include "torch/nvidia/c10.h"
#endif
#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
void AtenCast<kDev>::operator()(const Tensor input, Tensor out) const {
  TORCH_CHECK(supports(input, out), "Unsupported Cast metadata.");
  if (input.data() == out.data() && input.dtype() == out.dtype() &&
      input.strides() == out.strides()) {
    return;
  }
  const auto run = [&] {
    auto x = ToAtenTensor<kDev>(const_cast<void*>(input.data()), input.shape(),
                                input.strides(), input.dtype(),
                                input.device().index());
    auto y = ToAtenTensor<kDev>(out.data(), out.shape(), out.strides(),
                                out.dtype(), out.device().index());
    // Independently wrapped pointers have distinct ATen storage identities.
    const auto overlaps = [](const at::Tensor& a, const at::Tensor& b) {
      const auto ap = reinterpret_cast<std::uintptr_t>(a.data_ptr());
      const auto bp = reinterpret_cast<std::uintptr_t>(b.data_ptr());
      const auto an = a.storage().nbytes(), bn = b.storage().nbytes();
      return an && bn && (ap <= bp ? bp - ap < an : ap - bp < bn);
    };
    TORCH_CHECK(!overlaps(y, x), "Input and output storage must not overlap.");
    if (y.numel() != 0) {
      std::vector<int64_t> dimensions(y.dim());
      std::iota(dimensions.begin(), dimensions.end(), 0);
      std::sort(dimensions.begin(), dimensions.end(),
                [&](auto a, auto b) { return y.stride(a) < y.stride(b); });
      // The factory has already validated the complete storage byte span.
      std::size_t extent = 1;
      for (auto dim : dimensions) {
        if (y.size(dim) <= 1) continue;
        TORCH_CHECK(y.stride(dim) >= 0 &&
                        static_cast<std::size_t>(y.stride(dim)) >= extent,
                    "Output elements must not overlap.");
        extent += static_cast<std::size_t>(y.size(dim) - 1) *
                  static_cast<std::size_t>(y.stride(dim));
      }
    }
    y.copy_(x);
  };
  if constexpr (kDev == Device::Type::kNvidia) {
    const typename C10<kDev>::StreamGuard guard{
        C10<kDev>::GetStreamFromExternal(stream_, out.device().index())};
    run();
  } else {
    run();
  }
}

#ifdef WITH_CPU
template class AtenCast<Device::Type::kCpu>;
#endif
#ifdef WITH_NVIDIA
template class AtenCast<Device::Type::kNvidia>;
#endif

}  // namespace infini::ops
