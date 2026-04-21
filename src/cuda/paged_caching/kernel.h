#ifndef INFINI_OPS_CUDA_PAGED_CACHING_KERNEL_H_
#define INFINI_OPS_CUDA_PAGED_CACHING_KERNEL_H_

#include <cassert>
#include <cstdint>

#include "base/paged_caching.h"
#include "cuda/paged_caching/kernel.cuh"
#include "cuda/runtime_utils.h"
#include "data_type.h"
#include "dispatcher.h"

namespace infini::ops {

template <typename Backend>
class CudaPagedCaching : public PagedCaching {
 public:
  using PagedCaching::PagedCaching;

  void operator()(Tensor k_cache, Tensor v_cache, const Tensor k,
                  const Tensor v,
                  const Tensor slot_mapping) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    const auto num_tokens = slot_mapping_shape_[0];
    const auto num_kv_heads = k_shape_[1];
    const auto head_size = k_shape_[2];
    const auto block_size = k_cache_shape_[2];

    const auto k_src_stride = k_strides_[0];
    const auto v_src_stride = v_strides_[0];
    const auto k_cache_block_stride = k_cache_strides_[0];
    const auto v_cache_block_stride = v_cache_strides_[0];
    const auto k_cache_head_stride = k_cache_strides_[1];
    const auto v_cache_head_stride = v_cache_strides_[1];
    const auto k_cache_slot_stride = k_cache_strides_[2];
    const auto v_cache_slot_stride = v_cache_strides_[2];

    dim3 grid(static_cast<unsigned int>(num_kv_heads),
              static_cast<unsigned int>(num_tokens));

    int block_size_dim =
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    // The kernel is dispatched over the cache `dtype` (FP16 / BF16 / FP32)
    // and the runtime block size; `slot_mapping` must be 64-bit integer.
    assert(slot_mapping_type_ == DataType::kInt64 &&
           "`slot_mapping` must be `int64`");

    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>,
                 AllCudaBlockSizes>(
        {static_cast<int64_t>(k_cache_type_), block_size_dim},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          PagedCachingKernel<T, std::int64_t, kBlockSize>
              <<<grid, kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(k_cache.data()),
                  reinterpret_cast<T*>(v_cache.data()),
                  reinterpret_cast<const T*>(k.data()),
                  reinterpret_cast<const T*>(v.data()),
                  reinterpret_cast<const std::int64_t*>(slot_mapping.data()),
                  head_size, block_size, k_src_stride, v_src_stride,
                  k_cache_block_stride, v_cache_block_stride,
                  k_cache_head_stride, v_cache_head_stride,
                  k_cache_slot_stride, v_cache_slot_stride);
        },
        "CudaPagedCaching::operator()");
  }
};

}  // namespace infini::ops

#endif
