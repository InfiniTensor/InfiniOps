#ifndef INFINI_OPS_BASE_RESHAPE_AND_CACHE_FLASH_H_
#define INFINI_OPS_BASE_RESHAPE_AND_CACHE_FLASH_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

#include "data_type.h"
#include "operator.h"

namespace infini::ops {

// Aligned with vLLM `_custom_ops.reshape_and_cache_flash`.
class ReshapeAndCacheFlash : public Operator<ReshapeAndCacheFlash> {
 public:
  static bool supports(const Tensor& key, const Tensor& value,
                       const Tensor& slots, const Tensor& k_scale,
                       const Tensor& v_scale, const std::string& mode,
                       const Tensor& key_cache, const Tensor& value_cache) {
    return SupportsCommon(key, value, slots, k_scale, v_scale, mode, key_cache,
                          value_cache) &&
           // The base kernels share one `head_size` for key and value; only
           // providers whose kernels copy value with its own head size (the
           // CUDA-family template) accept the MLA latent pair, via their own
           // `supports` override.
           key.shape() == value.shape();
  }

  ReshapeAndCacheFlash(const Tensor key, const Tensor value,
                       const Tensor slot_mapping, const Tensor k_scale,
                       const Tensor v_scale, const std::string kv_cache_dtype,
                       Tensor key_cache, Tensor value_cache)
      : dtype_{key.dtype()},
        num_tokens_{slot_mapping.size(0)},
        num_heads_{key.size(1)},
        head_size_{key.size(2)},
        value_head_size_{value.size(2)},
        block_size_{key_cache.size(1)},
        key_token_stride_{key.stride(0)},
        value_token_stride_{value.stride(0)},
        key_head_stride_{key.stride(1)},
        value_head_stride_{value.stride(1)},
        key_cache_block_stride_{key_cache.stride(0)},
        value_cache_block_stride_{value_cache.stride(0)},
        key_cache_page_stride_{key_cache.stride(1)},
        value_cache_page_stride_{value_cache.stride(1)},
        key_cache_head_stride_{key_cache.stride(2)},
        value_cache_head_stride_{value_cache.stride(2)} {
    assert(key.ndim() == 3 && value.ndim() == 3 &&
           "`ReshapeAndCacheFlash` requires 3D `key` and `value`");
    assert((key.shape() == value.shape() ||
            (key.size(1) == value.size(1) && head_size_ == 576 &&
             value_head_size_ == 512)) &&
           key.dtype() == value.dtype() &&
           "`ReshapeAndCacheFlash` requires `key` and `value` to have matching "
           "shapes and dtypes, or the MLA latent pair");
    assert((dtype_ == DataType::kFloat16 || dtype_ == DataType::kBFloat16 ||
            dtype_ == DataType::kFloat32) &&
           "`ReshapeAndCacheFlash` supports float16, bfloat16, and float32");
    assert(slot_mapping.ndim() == 1 &&
           slot_mapping.dtype() == DataType::kInt64 &&
           slot_mapping.stride(0) == 1 &&
           "`ReshapeAndCacheFlash` requires contiguous int64 `slot_mapping`");
    assert(num_tokens_ <= key.size(0) && num_tokens_ <= value.size(0) &&
           num_heads_ == value.size(1) &&
           "`ReshapeAndCacheFlash` requires enough `key` and `value` rows and "
           "matching head counts");
    assert(num_heads_ > 0 && head_size_ > 0 && value_head_size_ > 0 &&
           "`ReshapeAndCacheFlash` requires non-empty head dimensions");
    // MLA caches a 576-dim latent key with a 512-dim value latent;
    // otherwise key and value share one head dimension.
    assert(value_head_size_ == head_size_ ||
           (head_size_ == 576 && value_head_size_ == 512));
    assert(key_cache.ndim() == 4 && value_cache.ndim() == 4 &&
           key_cache.size(0) == value_cache.size(0) &&
           key_cache.size(1) == value_cache.size(1) &&
           key_cache.size(2) == value_cache.size(2) &&
           "`ReshapeAndCacheFlash` requires matching 4D cache "
           "batch/page/head layout");
    assert(block_size_ > 0 && key_cache.size(2) == num_heads_ &&
           key_cache.size(3) == head_size_ &&
           value_cache.size(3) == value_head_size_ &&
           "`ReshapeAndCacheFlash` cache shape must be "
           "[`num_blocks`, `block_size`, `num_heads`, `head_size`]");
    assert(key.stride(2) == 1 && value.stride(2) == 1 &&
           key_cache.stride(3) == 1 && value_cache.stride(3) == 1 &&
           "`ReshapeAndCacheFlash` requires contiguous head dimensions");
    assert(key.device() == value.device() &&
           key.device() == slot_mapping.device() &&
           key.device() == k_scale.device() &&
           key.device() == v_scale.device() &&
           key.device() == key_cache.device() &&
           key.device() == value_cache.device() &&
           "`ReshapeAndCacheFlash` tensors must be on the same device");
    assert(k_scale.shape() == v_scale.shape() &&
           (k_scale.numel() == 1 || k_scale.numel() == num_heads_) &&
           k_scale.dtype() == DataType::kFloat32 &&
           v_scale.dtype() == DataType::kFloat32 &&
           "`ReshapeAndCacheFlash` scales must be float32 scalar or per-head "
           "tensors");
    assert(kv_cache_dtype == "auto" && key_cache.dtype() == dtype_ &&
           value_cache.dtype() == dtype_ &&
           "`ReshapeAndCacheFlash` currently supports `auto` cache dtype");
    assert(!key_cache.HasBroadcastDim() && !value_cache.HasBroadcastDim() &&
           "`ReshapeAndCacheFlash` caches must not have broadcast dimensions");
  }

  virtual void operator()(const Tensor key, const Tensor value,
                          const Tensor slot_mapping, const Tensor k_scale,
                          const Tensor v_scale,
                          const std::string kv_cache_dtype, Tensor key_cache,
                          Tensor value_cache) const = 0;

 protected:
  // The MLA packed latent layout: a 576-dim key (512 shared latent + 64 rope)
  // with a 512-dim value latent at matching head counts.
  static bool MlaLatentPair(const Tensor& key, const Tensor& value) {
    return key.size(1) == value.size(1) && key.size(2) == 576 &&
           value.size(2) == 512;
  }

  // Checks shared by every provider; layout acceptance is decided by the
  // caller so each implementation family gates what its kernel can copy.
  static bool SupportsCommon(const Tensor& key, const Tensor& value,
                             const Tensor& slots, const Tensor& k_scale,
                             const Tensor& v_scale, const std::string& mode,
                             const Tensor& key_cache,
                             const Tensor& value_cache) {
    if (key.ndim() != 3 || value.ndim() != 3 || slots.ndim() != 1 ||
        key_cache.ndim() != 4 || value_cache.ndim() != 4)
      return false;
    return key.dtype() == value.dtype() &&
           (key.dtype() == DataType::kFloat16 ||
            key.dtype() == DataType::kBFloat16 ||
            key.dtype() == DataType::kFloat32) &&
           slots.dtype() == DataType::kInt64 && slots.stride(0) == 1 &&
           slots.size(0) <= key.size(0) &&
           // The kernel indexes value with the same head/token grid as key.
           value.size(0) >= slots.size(0) && value.size(1) == key.size(1) &&
           key.size(1) > 0 && key.size(2) > 0 && value.size(2) > 0 &&
           key_cache.size(0) == value_cache.size(0) &&
           key_cache.size(1) == value_cache.size(1) &&
           key_cache.size(2) == value_cache.size(2) && key_cache.size(1) > 0 &&
           key_cache.size(2) == key.size(1) &&
           key_cache.size(3) == key.size(2) &&
           value_cache.size(3) == value.size(2) && key.stride(2) == 1 &&
           value.stride(2) == 1 && key_cache.stride(3) == 1 &&
           value_cache.stride(3) == 1 && mode == "auto" &&
           key_cache.dtype() == key.dtype() &&
           value_cache.dtype() == key.dtype() && !key_cache.HasBroadcastDim() &&
           !value_cache.HasBroadcastDim() &&
           k_scale.shape() == v_scale.shape() &&
           (k_scale.numel() == 1 || k_scale.numel() == key.size(1)) &&
           k_scale.dtype() == DataType::kFloat32 &&
           v_scale.dtype() == DataType::kFloat32;
  }

  DataType dtype_;

  std::size_t num_tokens_{0};

  std::size_t num_heads_{0};

  std::size_t head_size_{0};

  std::size_t value_head_size_{0};

  std::size_t block_size_{0};

  Tensor::Stride key_token_stride_{0};

  Tensor::Stride value_token_stride_{0};

  Tensor::Stride key_head_stride_{0};

  Tensor::Stride value_head_stride_{0};

  Tensor::Stride key_cache_block_stride_{0};

  Tensor::Stride value_cache_block_stride_{0};

  Tensor::Stride key_cache_page_stride_{0};

  Tensor::Stride value_cache_page_stride_{0};

  Tensor::Stride key_cache_head_stride_{0};

  Tensor::Stride value_cache_head_stride_{0};
};

}  // namespace infini::ops

#endif
