#ifndef INFINI_OPS_CUDA_FUSED_QK_NORM_KERNEL_H_
#define INFINI_OPS_CUDA_FUSED_QK_NORM_KERNEL_H_

#include <algorithm>
#include <cstdint>

#include "base/fused_qk_norm.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/fused_qk_norm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaFusedQkNorm : public FusedQkNorm {
 public:
  using FusedQkNorm::FusedQkNorm;

  void operator()(Tensor qkv, int64_t, int64_t, int64_t, int64_t, float,
                  Tensor q_weight, Tensor k_weight) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    const int block_size =
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    DispatchFunc<ReducedFloatTypes, AllCudaBlockSizes>(
        {static_cast<int64_t>(qkv.dtype()), static_cast<int64_t>(block_size)},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using TAcc = float;

          // One block per (token, head); V heads exit immediately.
          const auto total_heads =
              static_cast<unsigned>(num_heads_q_ + num_heads_k_ + num_heads_v_);
          const dim3 grid(total_heads,
                          static_cast<unsigned>(qkv_shape_.front()));
          constexpr int kBlockSize = ListGet<1>(list_tag);
          FusedQkNormKernel<kBlockSize, Backend::kDeviceType, T, TAcc>
              <<<grid, kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(qkv.data()),
                  static_cast<int64_t>(head_dim_),
                  static_cast<int64_t>(qkv.stride(0)),
                  reinterpret_cast<const T*>(q_weight.data()),
                  reinterpret_cast<const T*>(k_weight.data()),
                  static_cast<unsigned>(num_heads_q_),
                  static_cast<unsigned>(num_heads_k_),
                  static_cast<unsigned>(head_dim_), eps_);
        },
        "CudaFusedQkNorm::operator()");
  }
};

}  // namespace infini::ops

#endif
