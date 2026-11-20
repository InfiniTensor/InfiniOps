#ifndef INFINI_OPS_CUDA_CLOVER_LOCAL_MEMORY_KERNEL_H_
#define INFINI_OPS_CUDA_CLOVER_LOCAL_MEMORY_KERNEL_H_

#include "base/clover_local_memory.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/clover_local_memory/kernel.cuh"

namespace infini::ops {

template <typename Backend>
class CudaCloverLocalMemory : public CloverLocalMemory {
 public:
  using CloverLocalMemory::CloverLocalMemory;

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor g, const Tensor state, const Tensor indices,
                  const Tensor valid, const Tensor logit_scale,
                  const Tensor history_bias, const Tensor history_c,
                  size_t block_size, Tensor out) const override {
    const auto tokens = q.size(1);
    const auto requests = indices.numel();
    const auto num_v_heads = v.size(2);
    const auto value_dim = v.size(3);
    if (tokens == 0 || requests == 0 || num_v_heads == 0) return;

    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    constexpr int kVec = 8;
    dim3 grid(static_cast<unsigned>(value_dim / kVec),
              static_cast<unsigned>(requests),
              static_cast<unsigned>(num_v_heads));
    dim3 block(static_cast<unsigned>(block_size));
    const size_t shared = block_size * sizeof(float);

    const float scale = 1.0f / sqrtf(128.0f);
    const float reference = static_cast<float>(block_size);

    DispatchFunc<ReducedFloatTypes, List<DataType::kInt32, DataType::kInt64>>(
        {static_cast<int64_t>(q.dtype()),
         static_cast<int64_t>(indices.dtype())},
        [&](auto list_tag) {
          using Data = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using Index = TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
          CloverLocalMemoryKernel<Backend::kDeviceType, Data, Index>
              <<<grid, block, shared, cuda_stream>>>(
                  reinterpret_cast<const Data*>(q.data()),
                  reinterpret_cast<const Data*>(k.data()),
                  reinterpret_cast<const Data*>(v.data()),
                  reinterpret_cast<const float*>(g.data()),
                  reinterpret_cast<const float*>(state.data()),
                  reinterpret_cast<const Index*>(indices.data()),
                  reinterpret_cast<const int8_t*>(valid.data()),
                  reinterpret_cast<const Data*>(logit_scale.data()),
                  reinterpret_cast<const Data*>(history_bias.data()),
                  reinterpret_cast<const Data*>(history_c.data()),
                  reinterpret_cast<Data*>(out.data()),
                  static_cast<int64_t>(q.stride(1)),  // t (dim0 is batch=1)
                  static_cast<int64_t>(q.stride(2)),
                  static_cast<int64_t>(q.stride(3)),
                  static_cast<int64_t>(k.stride(1)),
                  static_cast<int64_t>(k.stride(2)),
                  static_cast<int64_t>(k.stride(3)),
                  static_cast<int64_t>(v.stride(1)),
                  static_cast<int64_t>(v.stride(2)),
                  static_cast<int64_t>(v.stride(3)),
                  static_cast<int64_t>(g.stride(1)),
                  static_cast<int64_t>(g.stride(2)),
                  static_cast<int64_t>(state.stride(0)),
                  static_cast<int64_t>(state.stride(1)),
                  static_cast<int64_t>(state.stride(2)),
                  static_cast<int64_t>(state.stride(3)),
                  static_cast<int64_t>(out.stride(1)),
                  static_cast<int64_t>(out.stride(2)),
                  static_cast<int64_t>(out.stride(3)),
                  scale, reference, static_cast<int>(block_size),
                  static_cast<int>(state.size(0)),
                  static_cast<int>(q.size(2)),
                  static_cast<int>(k.size(2)),
                  static_cast<int>(num_v_heads));
        },
        "CudaCloverLocalMemory::operator()");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_LOCAL_MEMORY_KERNEL_H_
