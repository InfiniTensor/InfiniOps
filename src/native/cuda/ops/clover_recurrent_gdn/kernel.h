#ifndef INFINI_OPS_CUDA_CLOVER_RECURRENT_GDN_KERNEL_H_
#define INFINI_OPS_CUDA_CLOVER_RECURRENT_GDN_KERNEL_H_

#include "base/clover_recurrent_gdn.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/clover_recurrent_gdn/kernel.cuh"

namespace infini::ops {

template <typename Backend>
class CudaCloverRecurrentGdn : public CloverRecurrentGdn {
 public:
  using CloverRecurrentGdn::CloverRecurrentGdn;

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor g, const Tensor beta, const Tensor offsets,
                  const Tensor initial_indices, const Tensor final_indices,
                  Tensor state, Tensor out) const override {
    // Layout [1, tokens, heads, dim]; strides skip the batch dim.
    const auto tokens = q.size(1);
    const auto num_heads = v.size(2);
    const auto value_dim = v.size(3);
    const auto requests = offsets.numel() - 1;
    if (tokens == 0 || requests <= 0) return;

    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    constexpr int kBV = 16;
    dim3 grid(static_cast<unsigned>(requests),
              static_cast<unsigned>(num_heads),
              static_cast<unsigned>((value_dim + kBV - 1) / kBV));
    dim3 block(static_cast<unsigned>(kBV));

    const int num_k_heads = static_cast<int>(k.size(2));

    DispatchFunc<ReducedFloatTypes>(
        {static_cast<int64_t>(q.dtype())},
        [&](auto list_tag) {
          using Data = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          CloverRecurrentGdnKernel<Backend::kDeviceType, Data>
              <<<grid, block, 0, cuda_stream>>>(
                  reinterpret_cast<const Data*>(q.data()),
                  reinterpret_cast<const Data*>(k.data()),
                  reinterpret_cast<const Data*>(v.data()),
                  reinterpret_cast<const float*>(g.data()),
                  reinterpret_cast<const float*>(beta.data()),
                  reinterpret_cast<const int32_t*>(offsets.data()),
                  reinterpret_cast<const int32_t*>(initial_indices.data()),
                  reinterpret_cast<const int32_t*>(final_indices.data()),
                  reinterpret_cast<float*>(state.data()),
                  reinterpret_cast<Data*>(out.data()),
                  static_cast<int64_t>(q.stride(1)),
                  static_cast<int64_t>(q.stride(2)),
                  static_cast<int64_t>(k.stride(1)),
                  static_cast<int64_t>(k.stride(2)),
                  static_cast<int64_t>(v.stride(1)),
                  static_cast<int64_t>(v.stride(2)),
                  static_cast<int64_t>(g.stride(1)),
                  static_cast<int64_t>(g.stride(2)),
                  static_cast<int64_t>(beta.stride(1)),
                  static_cast<int64_t>(beta.stride(2)),
                  static_cast<int64_t>(state.stride(0)),
                  static_cast<int64_t>(state.stride(1)),
                  static_cast<int64_t>(state.stride(2)),
                  static_cast<int64_t>(state.stride(3)),
                  static_cast<int64_t>(out.stride(1)),
                  static_cast<int64_t>(out.stride(2)),
                  static_cast<int64_t>(tokens),
                  static_cast<int64_t>(state.size(0)), num_k_heads,
                  static_cast<int>(value_dim), kBV);
        },
        "CudaCloverRecurrentGdn::operator()");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_RECURRENT_GDN_KERNEL_H_
