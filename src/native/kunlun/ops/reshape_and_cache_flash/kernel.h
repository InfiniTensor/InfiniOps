#ifndef INFINI_OPS_KUNLUN_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_KUNLUN_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/reshape_and_cache_flash.h"

// Vendor `libapiinfer.so` (Kunlun inference kernel library; shipped in the
// vLLM-XPU wheel, `vllm_xpu/so/`) exposes a paged KV-cache scatter with the
// same layout contract InfiniOps uses: key_cache/value_cache are
// `[num_blocks, block_size, num_heads, head_size]` and `slot_mapping` holds
// flat slots (`block_idx * block_size + offset`). The vendor header
// (`infer_ops.h`) is not installed alongside the library, so the entry point
// is declared here with the exact vendor template signature and resolved at
// link time. Three vendor specifics:
//   * It takes a `baidu::xpu::api::Context` — the "old" refactor context
//     (factory in `xpu/refactor/context/newcontext.h`, implemented in
//     `libxpuapi.so`), not the `xpytorch::xpu::api` one.
//   * The library only instantiates the vendor `float16`/`bfloat16` class
//     types (global-scope, declared by `xpu/refactor/util/{float16,
//     bfloat16}.h` via `core/dtype.h`), which are 16-bit bit-compatible with
//     fp16/bf16 storage, so payloads are passed through a reinterpret cast.
//   * The slot type is pinned to `int` (int32), while the InfiniOps contract
//     carries int64 slots — the wrapper narrows them on the host first.
#include <xpu/refactor/context/newcontext.h>
#include <xpu/refactor/core/dtype.h>

// The vendor entry point lives in `namespace infer_ops` (mangled exactly as
// `infer_ops::reshape_and_cache<T, TID>` in `libapiinfer.so`).
namespace infer_ops {

template <typename T, typename TID>
int reshape_and_cache(baidu::xpu::api::Context* ctx, const T* k, const T* v,
                      T* k_cache, T* v_cache, TID* slot_mapping,
                      std::int64_t num_tokens, std::int64_t num_blocks,
                      std::int64_t num_heads, std::int64_t k_head_size,
                      std::int64_t v_head_size, std::int64_t block_size);

}  // namespace infer_ops

namespace infini::ops {

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kKunlun>
    : public ReshapeAndCacheFlash {
 public:
  using ReshapeAndCacheFlash::ReshapeAndCacheFlash;

  bool capture_safe() const override { return true; }

  static bool supports(const Tensor& key, const Tensor& value,
                       const Tensor& slots, const Tensor& k_scale,
                       const Tensor& v_scale, const std::string& mode,
                       const Tensor& key_cache, const Tensor& value_cache) {
    // The base `SupportsCommon` already enforces the canonical FA layout
    // (`[blocks, block, heads, dim]`, contiguous, unit last stride,
    // int64 slots, float32 scales). Kunlun adds a dtype restriction to the
    // 16-bit cache types the vendor kernel instantiates.
    return SupportsCommon(key, value, slots, k_scale, v_scale, mode, key_cache,
                          value_cache) &&
           (key.dtype() == DataType::kBFloat16 ||
            key.dtype() == DataType::kFloat16);
  }

  void operator()(const Tensor key, const Tensor value, const Tensor slots,
                  const Tensor k_scale, const Tensor v_scale,
                  const std::string kv_cache_dtype, Tensor key_cache,
                  Tensor value_cache) const override {
    (void)k_scale;
    (void)v_scale;
    (void)kv_cache_dtype;
    if (num_tokens_ == 0) {
      return;
    }

    baidu::xpu::api::Context* ctx = VendorContext();

    // The vendor kernel reads int32 slots; the InfiniOps contract hands over
    // int64. Narrow through a host staging buffer (sizes here are bounded by
    // the submission chunk, so the round trip is negligible) and push the
    // compacted copy back to the device.
    thread_local std::vector<int> slots_host;
    slots_host.resize(num_tokens_);
    thread_local int* slots_dev = nullptr;
    thread_local std::int64_t slots_dev_cap = 0;
    if (slots_dev_cap < num_tokens_) {
      if (slots_dev) {
        xpu_free(slots_dev);
        slots_dev = nullptr;
      }
      slots_dev_cap = 0;
      void* raw = nullptr;
      if (xpu_malloc(&raw, sizeof(int) * num_tokens_) != 0) {
        throw std::runtime_error(
            "kunlun reshape_and_cache slot buffer "
            "allocation failed");
      }
      slots_dev = static_cast<int*>(raw);
      slots_dev_cap = num_tokens_;
    }
    thread_local std::vector<int64_t> slots64_host;
    slots64_host.resize(num_tokens_);
    xpu_wait(stream_);
    if (xpu_memcpy(slots64_host.data(), slots.data(),
                   num_tokens_ * sizeof(int64_t), XPU_DEVICE_TO_HOST) != 0) {
      throw std::runtime_error("kunlun reshape_and_cache slot download failed");
    }
    for (std::int64_t i = 0; i < num_tokens_; ++i) {
      slots_host[i] = static_cast<int>(slots64_host[i]);
    }
    if (xpu_memcpy(slots_dev, slots_host.data(), num_tokens_ * sizeof(int),
                   XPU_HOST_TO_DEVICE) != 0) {
      throw std::runtime_error("kunlun reshape_and_cache slot upload failed");
    }

    // The caller may hand over split views of the fused-QKV GEMM output
    // ([tokens, heads, dim] with a sequence stride spanning Q+K+V); the
    // vendor kernel reads dense rows, so compact non-dense inputs first.
    const auto dense_rows = [](const Tensor& t) {
      return t.stride(0) == t.size(1) * t.size(2) &&
             t.stride(1) == static_cast<int64_t>(t.size(2)) && t.stride(2) == 1;
    };
    const int64_t tokens = key.size(0);
    const int64_t kv_heads = key.size(1);
    const int64_t dim = key.size(2);
    const bool k_dense = dense_rows(key);
    const bool v_dense = dense_rows(value);
    thread_local std::vector<char> staging;
    thread_local void* staging_dev = nullptr;
    thread_local int64_t staging_cap = 0;
    const char* kp = static_cast<const char*>(key.data());
    const char* vp = static_cast<const char*>(value.data());
    if (!k_dense || !v_dense) {
      const int64_t row_bytes = kv_heads * dim * 2;  // bf16/fp16 payload
      const int64_t total = 2 * tokens * row_bytes;
      staging.resize(total);
      if (staging_cap < total) {
        if (staging_dev) xpu_free(staging_dev);
        staging_dev = nullptr;
        staging_cap = 0;
        void* raw = nullptr;
        if (xpu_malloc(&raw, total) != 0) {
          throw std::runtime_error(
              "kunlun reshape_and_cache staging "
              "allocation failed");
        }
        staging_dev = raw;
        staging_cap = total;
      }
      char* k_host = staging.data();
      char* v_host = staging.data() + tokens * row_bytes;
      xpu_wait(stream_);
      for (int64_t t = 0; t < tokens; ++t) {
        if (!k_dense) {
          xpu_memcpy(k_host + t * row_bytes,
                     kp + t * key.stride(0) * sizeof(uint16_t), row_bytes,
                     XPU_DEVICE_TO_HOST);
        }
        if (!v_dense) {
          xpu_memcpy(v_host + t * row_bytes,
                     vp + t * value.stride(0) * sizeof(uint16_t), row_bytes,
                     XPU_DEVICE_TO_HOST);
        }
      }
      if (xpu_memcpy(staging_dev, staging.data(), total, XPU_HOST_TO_DEVICE) !=
          0) {
        throw std::runtime_error(
            "kunlun reshape_and_cache staging upload failed");
      }
      if (!k_dense) kp = static_cast<const char*>(staging_dev);
      if (!v_dense)
        vp = static_cast<const char*>(staging_dev) + tokens * row_bytes;
    }

    // Bind the vendor context to the caller's stream: InfiniRT submits its
    // kernels on its own XRE stream, and this op must order against them.
    ctx->set_stream(stream_);

    int rc;
    if (key.dtype() == DataType::kBFloat16) {
      rc = infer_ops::reshape_and_cache<bfloat16, int>(
          ctx, reinterpret_cast<const bfloat16*>(kp),
          reinterpret_cast<const bfloat16*>(vp),
          static_cast<bfloat16*>(key_cache.data()),
          static_cast<bfloat16*>(value_cache.data()), slots_dev,
          static_cast<std::int64_t>(num_tokens_),
          static_cast<std::int64_t>(key_cache.size(0)),
          static_cast<std::int64_t>(num_heads_),
          static_cast<std::int64_t>(head_size_),
          static_cast<std::int64_t>(value_head_size_),
          static_cast<std::int64_t>(block_size_));
    } else {
      rc = infer_ops::reshape_and_cache<float16, int>(
          ctx, reinterpret_cast<const float16*>(kp),
          reinterpret_cast<const float16*>(vp),
          static_cast<float16*>(key_cache.data()),
          static_cast<float16*>(value_cache.data()), slots_dev,
          static_cast<std::int64_t>(num_tokens_),
          static_cast<std::int64_t>(key_cache.size(0)),
          static_cast<std::int64_t>(num_heads_),
          static_cast<std::int64_t>(head_size_),
          static_cast<std::int64_t>(value_head_size_),
          static_cast<std::int64_t>(block_size_));
    }
    if (rc != 0) {
      throw std::runtime_error("kunlun reshape_and_cache failed, rc=" +
                               std::to_string(rc));
    }
  }

 private:
  static baidu::xpu::api::Context* VendorContext() {
    thread_local baidu::xpu::api::Context* ctx = nullptr;
    if (ctx == nullptr) {
      ctx = baidu::xpu::api::create_context();
      // Vendor kernels launch on `ctx->xpu_stream`; leave it null so they
      // run on the device default stream (InfiniRT submissions synchronize
      // the default stream after each op on kunlun).
      ctx->set_stream(nullptr);
    }
    return ctx;
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_KUNLUN_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
