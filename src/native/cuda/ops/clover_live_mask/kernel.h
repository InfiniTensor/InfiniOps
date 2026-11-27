#ifndef INFINI_OPS_CUDA_CLOVER_LIVE_MASK_KERNEL_H_
#define INFINI_OPS_CUDA_CLOVER_LIVE_MASK_KERNEL_H_

#include "base/clover_live_mask.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/clover_live_mask/kernel.cuh"

namespace infini::ops {

template <typename Backend>
class CudaCloverLiveMask : public CloverLiveMask {
 public:
  using CloverLiveMask::CloverLiveMask;

  void operator()(const Tensor ids, Tensor previous, int64_t mask_id,
                  bool refresh, Tensor live) const override {
    const auto n = ids.numel();
    if (n == 0) return;
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    constexpr unsigned int kThreads = 128;
    const auto blocks = utils::CeilDiv(n, static_cast<Tensor::Size>(kThreads));

    DispatchFunc<List<DataType::kInt32, DataType::kInt64>>(
        {static_cast<int64_t>(ids.dtype())},
        [&](auto list_tag) {
          using Index = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          CloverLiveMaskKernel<Backend::kDeviceType, Index>
              <<<static_cast<unsigned>(blocks), kThreads, 0, cuda_stream>>>(
                  reinterpret_cast<const Index*>(ids.data()),
                  reinterpret_cast<Index*>(previous.data()),
                  static_cast<int64_t>(mask_id), refresh ? 1 : 0,
                  reinterpret_cast<int8_t*>(live.data()), n);
        },
        "CudaCloverLiveMask::operator()");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_LIVE_MASK_KERNEL_H_
