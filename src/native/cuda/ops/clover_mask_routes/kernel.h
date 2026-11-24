#ifndef INFINI_OPS_CUDA_CLOVER_MASK_ROUTES_KERNEL_H_
#define INFINI_OPS_CUDA_CLOVER_MASK_ROUTES_KERNEL_H_

#include "base/clover_mask_routes.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/clover_mask_routes/kernel.cuh"

namespace infini::ops {

template <typename Backend>
class CudaCloverMaskRoutes : public CloverMaskRoutes {
 public:
  using CloverMaskRoutes::CloverMaskRoutes;

  void operator()(const Tensor ids, const Tensor live, int64_t num_experts,
                  Tensor out) const override {
    const auto n = ids.numel();
    if (n == 0) return;
    const auto width = ids.size(1);
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    constexpr unsigned int kThreads = 256;
    const auto blocks = utils::CeilDiv(n, static_cast<Tensor::Size>(kThreads));
    // A row's width that is a power of two turns the row index into a
    // shift; any other width keeps the division.
    const int shift = utils::Log2IfPowerOfTwo(width);

    DispatchFunc<List<DataType::kInt32, DataType::kInt64>>(
        {static_cast<int64_t>(ids.dtype())},
        [&](auto list_tag) {
          using Index = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          const Index* id_data = reinterpret_cast<const Index*>(ids.data());
          const int8_t* live_data =
              reinterpret_cast<const int8_t*>(live.data());
          Index* out_data = reinterpret_cast<Index*>(out.data());
          const Index mask = static_cast<Index>(num_experts);
          if (shift >= 0) {
            CloverMaskRoutesKernel<Backend::kDeviceType, Index>
                <<<static_cast<unsigned>(blocks), kThreads, 0, cuda_stream>>>(
                    id_data, live_data, mask, shift, out_data, n);
          } else {
            CloverMaskRoutesByDivisionKernel<Backend::kDeviceType, Index>
                <<<static_cast<unsigned>(blocks), kThreads, 0, cuda_stream>>>(
                    id_data, live_data, mask, static_cast<int64_t>(width),
                    out_data, n);
          }
        },
        "CudaCloverMaskRoutes::operator()");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_MASK_ROUTES_KERNEL_H_
