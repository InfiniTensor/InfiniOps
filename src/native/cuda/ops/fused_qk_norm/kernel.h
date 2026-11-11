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

    DispatchFunc<ReducedFloatTypes>(
        {static_cast<int64_t>(qkv.dtype())},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using TAcc = float;

          // One block per (token, head); V heads exit immediately.
          const auto total_heads =
              static_cast<unsigned>(num_heads_q_ + num_heads_k_ + num_heads_v_);
          const dim3 grid(total_heads,
                          static_cast<unsigned>(qkv_shape_.front()));
          switch (block_size) {
            case 128:
              FusedQkNormKernel<128, Backend::kDeviceType, T, TAcc>
                  <<<grid, 128, 0, cuda_stream>>>(
                      reinterpret_cast<T*>(qkv.data()),
                      static_cast<int64_t>(head_dim_),
                      static_cast<int64_t>(
                          (num_heads_q_ + num_heads_k_ + num_heads_v_) *
                          head_dim_),
                      reinterpret_cast<const T*>(q_weight.data()),
                      reinterpret_cast<const T*>(k_weight.data()),
                      static_cast<unsigned>(num_heads_q_),
                      static_cast<unsigned>(num_heads_k_),
                      static_cast<unsigned>(head_dim_), eps_);
              break;
            case 256:
              FusedQkNormKernel<256, Backend::kDeviceType, T, TAcc>
                  <<<grid, 256, 0, cuda_stream>>>(
                      reinterpret_cast<T*>(qkv.data()),
                      static_cast<int64_t>(head_dim_),
                      static_cast<int64_t>(
                          (num_heads_q_ + num_heads_k_ + num_heads_v_) *
                          head_dim_),
                      reinterpret_cast<const T*>(q_weight.data()),
                      reinterpret_cast<const T*>(k_weight.data()),
                      static_cast<unsigned>(num_heads_q_),
                      static_cast<unsigned>(num_heads_k_),
                      static_cast<unsigned>(head_dim_), eps_);
              break;
            case 512:
              FusedQkNormKernel<512, Backend::kDeviceType, T, TAcc>
                  <<<grid, 512, 0, cuda_stream>>>(
                      reinterpret_cast<T*>(qkv.data()),
                      static_cast<int64_t>(head_dim_),
                      static_cast<int64_t>(
                          (num_heads_q_ + num_heads_k_ + num_heads_v_) *
                          head_dim_),
                      reinterpret_cast<const T*>(q_weight.data()),
                      reinterpret_cast<const T*>(k_weight.data()),
                      static_cast<unsigned>(num_heads_q_),
                      static_cast<unsigned>(num_heads_k_),
                      static_cast<unsigned>(head_dim_), eps_);
              break;
            case 1024:
              FusedQkNormKernel<1024, Backend::kDeviceType, T, TAcc>
                  <<<grid, 1024, 0, cuda_stream>>>(
                      reinterpret_cast<T*>(qkv.data()),
                      static_cast<int64_t>(head_dim_),
                      static_cast<int64_t>(
                          (num_heads_q_ + num_heads_k_ + num_heads_v_) *
                          head_dim_),
                      reinterpret_cast<const T*>(q_weight.data()),
                      reinterpret_cast<const T*>(k_weight.data()),
                      static_cast<unsigned>(num_heads_q_),
                      static_cast<unsigned>(num_heads_k_),
                      static_cast<unsigned>(head_dim_), eps_);
              break;
            case 2048:
              FusedQkNormKernel<2048, Backend::kDeviceType, T, TAcc>
                  <<<grid, 2048, 0, cuda_stream>>>(
                      reinterpret_cast<T*>(qkv.data()),
                      static_cast<int64_t>(head_dim_),
                      static_cast<int64_t>(
                          (num_heads_q_ + num_heads_k_ + num_heads_v_) *
                          head_dim_),
                      reinterpret_cast<const T*>(q_weight.data()),
                      reinterpret_cast<const T*>(k_weight.data()),
                      static_cast<unsigned>(num_heads_q_),
                      static_cast<unsigned>(num_heads_k_),
                      static_cast<unsigned>(head_dim_), eps_);
              break;
            default:
              FusedQkNormKernel<128, Backend::kDeviceType, T, TAcc>
                  <<<grid, 128, 0, cuda_stream>>>(
                      reinterpret_cast<T*>(qkv.data()),
                      static_cast<int64_t>(head_dim_),
                      static_cast<int64_t>(
                          (num_heads_q_ + num_heads_k_ + num_heads_v_) *
                          head_dim_),
                      reinterpret_cast<const T*>(q_weight.data()),
                      reinterpret_cast<const T*>(k_weight.data()),
                      static_cast<unsigned>(num_heads_q_),
                      static_cast<unsigned>(num_heads_k_),
                      static_cast<unsigned>(head_dim_), eps_);
              break;
          }
        },
        "CudaFusedQkNorm::operator()");
  }
};

}  // namespace infini::ops

#endif
