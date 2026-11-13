#ifndef INFINI_OPS_CUDA_FUSED_QK_RMS_NORM_KERNEL_H_
#define INFINI_OPS_CUDA_FUSED_QK_RMS_NORM_KERNEL_H_

#include <cstdint>

#include "base/fused_qk_rms_norm.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/fused_qk_rms_norm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaFusedQkRmsNorm : public FusedQkRmsNorm {
 public:
  using FusedQkRmsNorm::FusedQkRmsNorm;

  void operator()(Tensor q, Tensor kv, Tensor q_weight, Tensor kv_weight,
                  float eps, Tensor q_output, Tensor kv_output) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    const int block_size =
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    DispatchFunc<ReducedFloatTypes, AllCudaBlockSizes>(
        {static_cast<int64_t>(q.dtype()), static_cast<int64_t>(block_size)},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using TAcc = float;

          // One block per (head, token, {q|kv}); the z axis picks the
          // tensor so both are normalized in a single launch.
          const dim3 grid(std::max(q_shape_[1], kv_shape_[1]),
                          static_cast<unsigned>(q_shape_[0]), 2);
          constexpr int kBlockSize = ListGet<1>(list_tag);
          FusedQkRmsNormKernel<kBlockSize, Backend::kDeviceType, T, TAcc>
              <<<grid, kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<const T*>(q.data()),
                  reinterpret_cast<const T*>(kv.data()),
                  reinterpret_cast<const T*>(q_weight.data()),
                  reinterpret_cast<const T*>(kv_weight.data()),
                  reinterpret_cast<T*>(q_output.data()),
                  reinterpret_cast<T*>(kv_output.data()), eps,
                  static_cast<unsigned>(q_shape_[1]),
                  static_cast<unsigned>(kv_shape_[1]),
                  static_cast<unsigned>(q_shape_[2]),
                  static_cast<unsigned>(kv_shape_[2]),
                  static_cast<int64_t>(q.stride(0)),
                  static_cast<int64_t>(kv.stride(0)),
                  static_cast<int64_t>(q.stride(1)),
                  static_cast<int64_t>(kv.stride(1)),
                  static_cast<int64_t>(q_output.stride(0)),
                  static_cast<int64_t>(kv_output.stride(0)),
                  static_cast<int64_t>(q_output.stride(1)),
                  static_cast<int64_t>(kv_output.stride(1)));
        },
        "CudaFusedQkRmsNorm::operator()");
  }
};

}  // namespace infini::ops

#endif
