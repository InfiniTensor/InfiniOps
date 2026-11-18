#ifndef INFINI_OPS_KUNLUN_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_
#define INFINI_OPS_KUNLUN_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/flash_attn_with_kvcache.h"

// Vendor `libxpu_flash_attention.so` (torch_xmlir package) implements paged
// decode attention over a block-tabled KV cache — the same cache contract
// InfiniOps uses (`[num_blocks, block_size, num_kv_heads, head_size]`, int32
// block table, per-sequence context lengths). The vendor header
// (`xhpc/xfa/include/flash_api.h`) declares the entry as
// `baidu::xpu::xfa::paged_attention_xft<T, TKVCACHE, TO, TL, TGEMM0, TGEMM1,
// TEW, TID, v_transed, quant_mode>`; `libxpu_flash_attention.so` ships the
// bf16/fp16 instantiations this wrapper binds. The context is the same
// `baidu::xpu::api::Context` family used by `libapiinfer.so`.
#include <xpu/refactor/context/newcontext.h>
#include <xpu/refactor/core/dtype.h>
#include <xpu/xdnn_types.h>

namespace baidu {
namespace xpu {
namespace xfa {

template <typename T, typename TKVCACHE, typename TO, typename TL,
          typename TGEMM0, typename TGEMM1, typename TEW, typename TID,
          bool v_transed, int quant_mode>
int paged_attention_xft(
    baidu::xpu::api::Context* ctx, TO* out, const T* query, const T* key,
    const T* value, const TKVCACHE* key_cache, const TKVCACHE* value_cache,
    std::int64_t num_kv_heads, float scale, const TID* block_table,
    const baidu::xpu::api::VectorParam<TID>& context_len,
    const baidu::xpu::api::VectorParam<TID>& valid_batch,
    std::int64_t block_size, std::int64_t max_context_len,
    const float* alibi_slopes, std::int64_t batch_num, std::int64_t head_num,
    std::int64_t head_dim, std::int64_t max_num_blocks_per_seq, const TO* shift,
    const TO* smooth, const float* query_maxptr, const float* key_cache_maxptr,
    const float* value_cache_maxptr, const T* key_cache_scales,
    const T* key_cache_zeros, const T* value_cache_scales,
    const T* value_cache_zeros, float* out_maxptr, std::int64_t v_head_dim);

}  // namespace xfa
}  // namespace xpu
}  // namespace baidu

namespace infini::ops {

// The vendor ships `paged_attention_xft` instantiations pinned to specific
// accumulate types: `tfloat32` for bfloat16, `float16` (matching the data
// type) for float16. The trait maps our payload type onto the pairing the
// library actually exports.
// The vendor pins the accumulate types per data type: for bfloat16 the
// local accumulator TL is `float` with `tfloat32` GEMMs; for float16 all
// three are `float16` (TEW stays float in both).
template <typename T>
struct GemmAccum;
template <>
struct GemmAccum<bfloat16> {
  using type = tfloat32;
};
template <>
struct GemmAccum<float16> {
  using type = float16;
};

template <typename T>
struct LocalAccum;
template <>
struct LocalAccum<bfloat16> {
  using type = float;
};
template <>
struct LocalAccum<float16> {
  using type = float16;
};

template <>
class Operator<FlashAttnWithKvcache, Device::Type::kKunlun>
    : public FlashAttnWithKvcache {
 public:
  using FlashAttnWithKvcache::FlashAttnWithKvcache;

  static bool supports_impl(const Tensor q, Tensor k_cache, Tensor v_cache,
                            const std::optional<Tensor> k,
                            const std::optional<Tensor> v,
                            const std::optional<Tensor> rotary_cos,
                            const std::optional<Tensor> rotary_sin,
                            const std::optional<Tensor> cache_seqlens,
                            const std::optional<Tensor> cache_batch_idx,
                            const std::optional<Tensor> cache_leftpad,
                            const std::optional<Tensor> block_table,
                            const std::optional<Tensor> alibi_slopes,
                            const std::optional<double> softmax_scale,
                            const bool causal,
                            const std::vector<int64_t> window_size,
                            const double softcap, const bool rotary_interleaved,
                            const int64_t num_splits,
                            const bool return_softmax_lse, Tensor out,
                            std::optional<Tensor> softmax_lse) {
    // The vendor kernel takes none of the optional extensions below;
    // reject them at plan time instead of call time — the same
    // supports-precision the nvidia provider applies. `causal` is not
    // gated: the vendor kernel has no causal flag, and with one query
    // token per sequence (enforced below) attention over the cache is
    // causal regardless, so either flag computes the same result.
    if (k || v || rotary_cos || rotary_sin || cache_batch_idx ||
        cache_leftpad || alibi_slopes ||
        window_size != std::vector<int64_t>{-1, -1} || softcap != 0.0 ||
        num_splits != 0 || return_softmax_lse || softmax_lse ||
        !cache_seqlens) {
      return false;
    }
    if (!FlashAttnWithKvcache::supports(
            q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
            cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
            softmax_scale, causal, window_size, softcap, rotary_interleaved,
            num_splits, return_softmax_lse, out, softmax_lse)) {
      return false;
    }
    // Paged decode only: the vendor kernel reads KV exclusively through
    // the block table and only supports its pinned page sizes; the static
    // backend's whole-slot pseudo pages (block_size == cache capacity,
    // e.g. 4096) keep their unfused path. The vendor kernel takes no
    // layout arguments, so the cache blocks must be fully dense
    // (the base only pins the last stride).
    const bool paged = block_table.has_value();
    const int64_t page_size = static_cast<int64_t>(k_cache.size(1));
    const bool cache_dense = k_cache.stride(2) == k_cache.size(3) &&
                             v_cache.stride(2) == v_cache.size(3);
    // The vendor kernel writes a dense [batch, 1, heads, dim] output; a
    // strided output view would be silently misplaced.
    const bool out_dense = out.stride(1) == out.size(2) * out.size(3) &&
                           out.stride(2) == static_cast<int64_t>(out.size(3));
    return paged && cache_dense && out_dense && q.size(1) == 1 &&
           page_size > 0 && page_size <= 512 &&
           (q.dtype() == DataType::kBFloat16 ||
            q.dtype() == DataType::kFloat16);
  }

  static bool supports(const Tensor q, Tensor k_cache, Tensor v_cache,
                       const std::optional<Tensor> k,
                       const std::optional<Tensor> v,
                       const std::optional<Tensor> rotary_cos,
                       const std::optional<Tensor> rotary_sin,
                       const std::optional<Tensor> cache_seqlens,
                       const std::optional<Tensor> cache_batch_idx,
                       const std::optional<Tensor> cache_leftpad,
                       const std::optional<Tensor> block_table,
                       const std::optional<Tensor> alibi_slopes,
                       const std::optional<double> softmax_scale,
                       const bool causal,
                       const std::vector<int64_t> window_size,
                       const double softcap, const bool rotary_interleaved,
                       const int64_t num_splits, const bool return_softmax_lse,
                       Tensor out, std::optional<Tensor> softmax_lse) {
    try {
      const bool ok = supports_impl(
          q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
          cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
          softmax_scale, causal, window_size, softcap, rotary_interleaved,
          num_splits, return_softmax_lse, out, softmax_lse);
      return ok;
    } catch (const std::exception& e) {
      throw;
    }
  }

  static bool supports(const Tensor q, Tensor k_cache, Tensor v_cache,
                       const std::optional<Tensor> k,
                       const std::optional<Tensor> v,
                       const std::optional<Tensor> rotary_cos,
                       const std::optional<Tensor> rotary_sin,
                       const int64_t cache_seqlens,
                       const std::optional<Tensor> cache_batch_idx,
                       const std::optional<Tensor> cache_leftpad,
                       const std::optional<Tensor> block_table,
                       const std::optional<Tensor> alibi_slopes,
                       const std::optional<double> softmax_scale,
                       const bool causal,
                       const std::vector<int64_t> window_size,
                       const double softcap, const bool rotary_interleaved,
                       const int64_t num_splits, const bool return_softmax_lse,
                       Tensor out, std::optional<Tensor> softmax_lse) {
    return supports(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin,
                    std::optional<Tensor>{}, cache_batch_idx, cache_leftpad,
                    block_table, alibi_slopes, softmax_scale, causal,
                    window_size, softcap, rotary_interleaved, num_splits,
                    return_softmax_lse, out, softmax_lse);
  }

  // The base declares one virtual overload taking a scalar
  // `cache_seqlens` and one taking an optional tensor; both must be
  // overridden, otherwise the class stays abstract and dispatch silently
  // skips it. The scalar form forwards to the tensor form.
  void operator()(const Tensor q, Tensor k_cache, Tensor v_cache,
                  const std::optional<Tensor> k, const std::optional<Tensor> v,
                  const std::optional<Tensor> rotary_cos,
                  const std::optional<Tensor> rotary_sin,
                  const int64_t cache_seqlens,
                  const std::optional<Tensor> cache_batch_idx,
                  const std::optional<Tensor> cache_leftpad,
                  const std::optional<Tensor> block_table,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool rotary_interleaved, const int64_t num_splits,
                  const bool return_softmax_lse, Tensor out,
                  std::optional<Tensor> softmax_lse) const override {
    (*this)(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin,
            std::optional<Tensor>{}, cache_batch_idx, cache_leftpad,
            block_table, alibi_slopes, softmax_scale, causal, window_size,
            softcap, rotary_interleaved, num_splits, return_softmax_lse, out,
            softmax_lse);
  }

  void operator()(const Tensor q, Tensor k_cache, Tensor v_cache,
                  const std::optional<Tensor> k, const std::optional<Tensor> v,
                  const std::optional<Tensor> rotary_cos,
                  const std::optional<Tensor> rotary_sin,
                  const std::optional<Tensor> cache_seqlens,
                  const std::optional<Tensor> cache_batch_idx,
                  const std::optional<Tensor> cache_leftpad,
                  const std::optional<Tensor> block_table,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool rotary_interleaved, const int64_t num_splits,
                  const bool return_softmax_lse, Tensor out,
                  std::optional<Tensor> softmax_lse) const override {
    Validate(k, v, rotary_cos, rotary_sin, cache_batch_idx, cache_leftpad,
             alibi_slopes, window_size, softcap, num_splits, return_softmax_lse,
             softmax_lse);

    // [batch, 1, heads, dim] vendor BLHD decode layout.
    const int64_t batch = q.size(0);
    const int64_t head_num = q.size(2);
    const int64_t head_dim = q.size(3);
    const int64_t kv_head_num = k_cache.size(2);
    const int64_t block_size = k_cache.size(1);
    const float scale = static_cast<float>(
        softmax_scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim))));

    baidu::xpu::api::Context* ctx = VendorContext();
    // Null `valid_batch` means one query per batch entry (one-to-one
    // mapping); InfiniLM decode steps submit exactly that, so the kernel
    // gets an empty vector there.

    int rc;
    if (q.dtype() == DataType::kBFloat16) {
      rc = Run<bfloat16>(ctx, stream_, q, k_cache, v_cache, cache_seqlens,
                         block_table, out, batch, head_num, kv_head_num,
                         head_dim, block_size, scale);
    } else {
      rc = Run<float16>(ctx, stream_, q, k_cache, v_cache, cache_seqlens,
                        block_table, out, batch, head_num, kv_head_num,
                        head_dim, block_size, scale);
    }
    if (rc != 0) {
      throw std::runtime_error("kunlun paged_attention_xft failed, rc=" +
                               std::to_string(rc));
    }
  }

 private:
  template <typename T>
  static int Run(baidu::xpu::api::Context* ctx, void* stream, const Tensor& q,
                 const Tensor& k_cache, const Tensor& v_cache,
                 const std::optional<Tensor>& cache_seqlens,
                 const std::optional<Tensor>& block_table, const Tensor& out,
                 int64_t batch, int64_t head_num, int64_t kv_head_num,
                 int64_t head_dim, int64_t block_size, float scale) {
    // The vendor kernel reads a dense [batch, 1, heads, dim] query. The
    // caller may pass a split view of the fused-QKV output whose sequence
    // stride spans Q+K+V, so compact non-dense inputs first.
    thread_local std::vector<T> q_dense;
    const T* query = static_cast<const T*>(q.data());
    if (q.stride(1) != q.size(2) * q.size(3) ||
        q.stride(2) != static_cast<int64_t>(q.size(3)) || q.stride(3) != 1) {
      const int64_t batch = q.size(0);
      const int64_t heads = q.size(2);
      const int64_t dim = q.size(3);
      // The row copy below assumes a unit inner stride; the base contract
      // (`q.stride(-1) == 1`) guarantees it, but fail loudly if a direct
      // caller ever violates it.
      if (q.stride(3) != 1) {
        return -1005;
      }
      q_dense.resize(batch * heads * dim);
      xpu_wait(stream);
      for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads; ++h) {
          if (xpu_memcpy(q_dense.data() + (b * heads + h) * dim,
                         static_cast<const char*>(q.data()) +
                             (b * q.stride(0) + h * q.stride(2)) * sizeof(T),
                         dim * sizeof(T), XPU_DEVICE_TO_HOST) != 0) {
            return -1003;
          }
        }
      }
      thread_local T* q_dense_dev = nullptr;
      thread_local int64_t q_dense_cap = 0;
      if (q_dense_cap < batch * heads * dim) {
        if (q_dense_dev) xpu_free(q_dense_dev);
        q_dense_dev = nullptr;
        q_dense_cap = 0;
        void* raw = nullptr;
        if (xpu_malloc(&raw, batch * heads * dim * sizeof(T)) != 0) {
          return -1002;
        }
        q_dense_dev = static_cast<T*>(raw);
        q_dense_cap = batch * heads * dim;
      }
      if (xpu_memcpy(q_dense_dev, q_dense.data(),
                     batch * heads * dim * sizeof(T),
                     XPU_HOST_TO_DEVICE) != 0) {
        return -1004;
      }
      query = q_dense_dev;
    }
    const auto* key_cache = static_cast<const T*>(k_cache.data());
    const auto* value_cache = static_cast<const T*>(v_cache.data());
    auto* o = static_cast<T*>(const_cast<void*>(out.data()));
    const int* block_table_ptr =
        block_table ? static_cast<const int*>(block_table->data()) : nullptr;
    // Per-sequence context lengths: the vendor reads `VectorParam.cpu` on
    // the host for launch planning and `VectorParam.xpu` on-card, so
    // materialize the host copy via the vendor runtime's device-to-host
    // helper.
    thread_local std::vector<int> context_len_host;
    const int* context_len_dev =
        cache_seqlens ? static_cast<const int*>(cache_seqlens->data())
                      : nullptr;
    const int64_t context_len_count =
        cache_seqlens ? static_cast<int64_t>(cache_seqlens->size(0)) : batch;
    if (context_len_dev) {
      context_len_host.assign(context_len_count, 0);
      xpu_wait(stream);
      if (xpu_memcpy(context_len_host.data(), context_len_dev,
                     context_len_host.size() * sizeof(int),
                     XPU_DEVICE_TO_HOST) != 0) {
        return -1000;
      }
    }
    // The kernel reads launch-planning lengths on the host (`cpu`) and
    // per-step values on-card (`xpu`) — the convention that makes the
    // vendor kernels run; keep both copies in sync via the D2H above.
    int* context_len_host_ptr =
        context_len_dev ? context_len_host.data() : nullptr;
    baidu::xpu::api::VectorParam<int> context_len{
        context_len_host_ptr, context_len_count,
        context_len_dev ? const_cast<int*>(context_len_dev) : nullptr};

    const int64_t max_context_len =
        block_table ? block_table->size(1) * block_size : k_cache.size(1);
    const int64_t max_blocks_per_seq = block_table ? block_table->size(1) : 1;

    // Order the vendor kernel against the caller's InfiniRT stream.
    ctx->set_stream(stream);

    int rc = baidu::xpu::xfa::paged_attention_xft<
        T, T, T, typename LocalAccum<T>::type, typename GemmAccum<T>::type,
        typename GemmAccum<T>::type, float, int, /*v_transed=*/false,
        /*quant_mode=*/0>(
        ctx, o, query,
        /*key=*/nullptr, /*value=*/nullptr, key_cache, value_cache,
        static_cast<std::int64_t>(kv_head_num), scale, block_table_ptr,
        context_len, /*valid_batch=*/baidu::xpu::api::VectorParam<int>{},
        static_cast<std::int64_t>(block_size), max_context_len,
        /*alibi_slopes=*/nullptr, static_cast<std::int64_t>(batch),
        static_cast<std::int64_t>(head_num),
        static_cast<std::int64_t>(head_dim), max_blocks_per_seq,
        /*shift=*/nullptr, /*smooth=*/nullptr, /*query_maxptr=*/nullptr,
        /*key_cache_maxptr=*/nullptr, /*value_cache_maxptr=*/nullptr,
        /*key_cache_scales=*/nullptr, /*key_cache_zeros=*/nullptr,
        /*value_cache_scales=*/nullptr, /*value_cache_zeros=*/nullptr,
        /*out_maxptr=*/nullptr, /*v_head_dim=*/-1);
    return rc;
  }

  static void Validate(const std::optional<Tensor>& k,
                       const std::optional<Tensor>& v,
                       const std::optional<Tensor>& rotary_cos,
                       const std::optional<Tensor>& rotary_sin,
                       const std::optional<Tensor>& cache_batch_idx,
                       const std::optional<Tensor>& cache_leftpad,
                       const std::optional<Tensor>& alibi_slopes,
                       const std::vector<int64_t>& window_size, double softcap,
                       int64_t num_splits, bool return_softmax_lse,
                       const std::optional<Tensor>& softmax_lse) {
    if (k.has_value() || v.has_value() || rotary_cos.has_value() ||
        rotary_sin.has_value() || cache_batch_idx.has_value() ||
        cache_leftpad.has_value() || alibi_slopes.has_value() ||
        window_size != std::vector<int64_t>{-1, -1} || softcap != 0.0 ||
        num_splits != 0 || return_softmax_lse || softmax_lse.has_value()) {
      throw std::invalid_argument(
          "kunlun FlashAttnWithKvcache received an unsupported option");
    }
  }

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

#endif  // INFINI_OPS_KUNLUN_FLASH_ATTN_WITH_KVCACHE_KERNEL_H_
