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

    DispatchFunc<List<DataType::kInt32, DataType::kInt64>>(
        {static_cast<int64_t>(ids.dtype())},
        [&](auto list_tag) {
          using Index = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          CloverMaskRoutesKernel<Backend::kDeviceType, Index>
              <<<static_cast<unsigned>(blocks), kThreads, 0, cuda_stream>>>(
                  reinterpret_cast<const Index*>(ids.data()),
                  reinterpret_cast<const int8_t*>(live.data()),
                  num_experts, static_cast<int64_t>(width),
                  reinterpret_cast<Index*>(out.data()), n);
        },
        "CudaCloverMaskRoutes::operator()");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_MASK_ROUTES_KERNEL_H_
