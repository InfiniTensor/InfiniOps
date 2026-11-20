#ifndef INFINI_OPS_CUDA_CLOVER_MERGE_MOE_KERNEL_H_
#define INFINI_OPS_CUDA_CLOVER_MERGE_MOE_KERNEL_H_

#include "base/clover_merge_moe.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/clover_merge_moe/kernel.cuh"

namespace infini::ops {

template <typename Backend>
class CudaCloverMergeMoe : public CloverMergeMoe {
 public:
  using CloverMergeMoe::CloverMergeMoe;

  void operator()(const Tensor fresh, const Tensor live, Tensor cache,
                  Tensor out) const override {
    const auto n = fresh.numel();
    if (n == 0) return;
    const auto width = fresh.size(1);
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    constexpr unsigned int kThreads = 256;
    const auto blocks = utils::CeilDiv(n, static_cast<Tensor::Size>(kThreads));

    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>>(
        {static_cast<int64_t>(fresh.dtype())},
        [&](auto list_tag) {
          using Data = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          CloverMergeMoeKernel<Backend::kDeviceType, Data>
              <<<static_cast<unsigned>(blocks), kThreads, 0, cuda_stream>>>(
                  reinterpret_cast<const Data*>(fresh.data()),
                  reinterpret_cast<const int8_t*>(live.data()),
                  reinterpret_cast<Data*>(cache.data()),
                  reinterpret_cast<Data*>(out.data()),
                  static_cast<int64_t>(width), n);
        },
        "CudaCloverMergeMoe::operator()");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_MERGE_MOE_KERNEL_H_
