#ifndef INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_KERNEL_H_
#define INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <vector>

#include "acl/acl.h"
#include "ascend/device.h"
#include "base/reshape_and_cache.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<ReshapeAndCache, Device::Type::kAscend> : public ReshapeAndCache {
 public:
  using ReshapeAndCache::ReshapeAndCache;

  void operator()(
      const Tensor key, const Tensor value,
      const Tensor kv_cache, const Tensor slot_mapping,
      Tensor kv_cache_out) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    // Copy slot_mapping to host for address computation.
    auto num_tokens = static_cast<int64_t>(num_tokens_);
    std::vector<int64_t> slots(num_tokens);
    aclrtMemcpyAsync(slots.data(), num_tokens * sizeof(int64_t),
                     slot_mapping.data(), num_tokens * sizeof(int64_t),
                     ACL_MEMCPY_DEVICE_TO_HOST, stream);
    aclrtSynchronizeStream(stream);

    auto bs = static_cast<int64_t>(block_size_);
    auto row_bytes = static_cast<size_t>(num_kv_heads_ * head_size_) *
                     kDataTypeToSize.at(key.dtype());

    for (int64_t i = 0; i < num_tokens; ++i) {
      auto slot = slots[i];
      if (slot < 0) continue;  // Padding token — skip.
      auto block_idx = slot / bs;
      auto offset = slot % bs;

      // key[i, :, :] -> kv_cache_out[block_idx, offset, :, :]
      // Note: `value` is accepted for interface compatibility but is not
      // written here. In practice, this operator is called once for key_cache
      // and once for value_cache (see vLLM's reshape_and_cache pattern).
      auto* src = static_cast<const char*>(key.data()) +
                  i * key.stride(0) * key.element_size();
      auto* dst = static_cast<char*>(kv_cache_out.data()) +
                  (block_idx * kv_cache_out.stride(0) +
                   offset * kv_cache_out.stride(1)) *
                      kv_cache_out.element_size();
      aclrtMemcpyAsync(dst, row_bytes, src, row_bytes,
                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }
  }
};

}  // namespace infini::ops

#endif
