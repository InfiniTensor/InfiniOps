#ifndef INFINI_OPS_CUDA_RESHAPE_AND_CACHE_KERNEL_H_
#define INFINI_OPS_CUDA_RESHAPE_AND_CACHE_KERNEL_H_

#include <cstddef>
#include <cstdint>

#include "base/reshape_and_cache.h"
#include "common/generic_utils.h"
#include "cuda/kernel_commons.cuh"
#include "cuda/reshape_and_cache/kernel.cuh"
#include "cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaReshapeAndCache : public ReshapeAndCache {
 public:
  CudaReshapeAndCache(const Tensor key, const Tensor value,
                      const Tensor kv_cache, const Tensor slot_mapping,
                      Tensor kv_cache_out)
      : ReshapeAndCache{key, value, kv_cache, slot_mapping, kv_cache_out} {}

  void operator()(const Tensor key, const Tensor value, const Tensor kv_cache,
                  const Tensor slot_mapping,
                  Tensor kv_cache_out) const override {
    int block_size_cfg =
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    DispatchFunc<AllFloatTypes, AllCudaBlockSizes>(
        {static_cast<int64_t>(key_dtype_), block_size_cfg},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          auto cuda_stream =
              static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

          // One thread block per token.
          dim3 gridDims(num_tokens_);
          dim3 blockDims(std::min(static_cast<Tensor::Size>(block_size_cfg),
                                  num_kv_heads_ * head_size_));

          const T* d_key = reinterpret_cast<const T*>(key.data());
          const T* d_value = reinterpret_cast<const T*>(value.data());
          T* d_kv_cache_out = reinterpret_cast<T*>(kv_cache_out.data());
          const int64_t* d_slot_mapping =
              reinterpret_cast<const int64_t*>(slot_mapping.data());

          const size_t num_blocks = kv_cache_shape_[1];

          ReshapeAndCacheKernel<T, kBlockSize>
              <<<gridDims, blockDims, 0, cuda_stream>>>(
                  d_key, d_value, d_kv_cache_out, d_slot_mapping, num_kv_heads_,
                  head_size_, block_size_, num_blocks);
        },
        "CudaReshapeAndCache::operator()");
  }
};

}  // namespace infini::ops

#endif
