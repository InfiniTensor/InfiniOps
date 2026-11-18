#ifndef INFINI_OPS_KUNLUN_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
#define INFINI_OPS_KUNLUN_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/flash_attn_varlen_func.h"

// Vendor `libxpu_flash_attention.so` (torch_xmlir package) implements varlen
// prefill attention (`baidu::xpu::xfa::flash_attention_context_vllm`): q is
// the packed query tokens (`[total_tokens, heads, dim]`, BLHD), per-sequence
// lengths travel as prefix sums (`cu_seqlens` style) in `VectorParam` lod
// vectors — `cpu` points at a host copy (launch planning) and `xpu` at the
// on-card copy (in-kernel reads), exactly how vLLM-XPU passes both copies.
// The bf16/fp16 instantiations ship in the library; the declaration below
// mirrors `xhpc/xfa/include/flash_api.h`.
#include <xpu/refactor/context/newcontext.h>
#include <xpu/refactor/core/dtype.h>
#include <xpu/xdnn_types.h>

namespace baidu {
namespace xpu {
namespace xfa {

template <typename T_Q, typename T_K, typename T_V, typename T_BIAS,
          typename T_O, typename T_GEMM0, typename T_GEMM1, typename T_EW,
          typename TID, int64_t quant_mode>
int flash_attention_context_vllm(
    baidu::xpu::api::Context* ctx, const T_Q* q, const T_K* k, const T_V* v,
    const T_BIAS* mask_bias, T_O* o, std::int64_t batch_num,
    std::int64_t max_seq_q, std::int64_t max_seq_kv, std::int64_t head_num,
    std::int64_t head_dim, std::int64_t kv_head_num, const float* alibi_slopes,
    const baidu::xpu::api::VectorParam<TID>& qlod,
    const baidu::xpu::api::VectorParam<TID>& kvlod,
    const std::vector<std::int64_t>& mask_bias_shape,
    const std::vector<std::int64_t>& alibi_slopes_shape,
    const bool is_causal_mask, const bool is_qkv_fusion,
    std::int64_t qkv_layout, std::int64_t vsl_type, const float* q_maxptr,
    const float* k_maxptr, const float* v_maxptr, float* o_maxptr,
    std::int64_t vo_head_dim, const T_K* k_cache, const T_V* v_cache,
    const float* k_cache_maxptr, const float* v_cache_maxptr,
    std::int64_t block_size, std::int64_t max_blocks_per_seq,
    std::int64_t prefill_len, const TID* block_tables, float alpha,
    std::int64_t swa_left_size, std::int64_t swa_right_size, T_EW* softmax_lse,
    bool unpadded_lse);

}  // namespace xfa
}  // namespace xpu
}  // namespace baidu

namespace infini::ops {

// The vendor ships `flash_attention_context_vllm` instantiations pinned to
// specific accumulate types: `float` GEMMs for bfloat16, `float16` GEMMs for
// float16 (both with a float bias/output-embed width). The trait maps our
// payload type onto the pairing the library actually exports.
template <typename T>
struct CtxVllmGemm;
template <>
struct CtxVllmGemm<bfloat16> {
  using type = float;
};
template <>
struct CtxVllmGemm<float16> {
  using type = float16;
};

template <typename T>
struct CtxVllmBias;
template <>
struct CtxVllmBias<bfloat16> {
  using type = float;
};
template <>
struct CtxVllmBias<float16> {
  using type = float16;
};

template <>
class Operator<FlashAttnVarlenFunc, Device::Type::kKunlun>
    : public FlashAttnVarlenFunc {
 public:
  using FlashAttnVarlenFunc::FlashAttnVarlenFunc;

  static bool supports(
      const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
      const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
      const int64_t max_seqlen_k, const double dropout_p,
      const std::optional<double> softmax_scale, const bool causal,
      const std::vector<int64_t> window_size, const double softcap,
      const bool deterministic, const bool return_attn_probs, Tensor out,
      std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) {
    // Varlen prefill over the paged KV cache only (block table + 4D
    // caches, the same contract as vLLM prefill). Dense varlen keeps the
    // caller's unfused path, which carries fp32-accumulation tuning the
    // vendor kernel does not provide.
    if (!block_table) {
      return false;
    }
    // Mirror the option guards enforced in `operator()` so unsupported
    // requests fail at plan time instead of call time — the same
    // supports-precision the nvidia provider applies.
    if (alibi_slopes || dropout_p != 0.0 || softcap != 0.0 || deterministic ||
        return_attn_probs || softmax_lse || s_dmask ||
        window_size != std::vector<int64_t>{-1, -1}) {
      return false;
    }
    if (!FlashAttnVarlenFunc::supports(
            q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, block_table,
            max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
            window_size, softcap, deterministic, return_attn_probs, out,
            softmax_lse, s_dmask)) {
      return false;
    }
    // The vendor kernel writes a dense packed output; a strided output view
    // would be silently misplaced (the base only pins the last stride).
    const bool out_dense = out.stride(0) == out.size(1) * out.size(2) &&
                           out.stride(1) == static_cast<int64_t>(out.size(2));
    return out_dense && (q.dtype() == DataType::kBFloat16 ||
                         q.dtype() == DataType::kFloat16);
  }

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                  const std::optional<Tensor> alibi_slopes,
                  const std::optional<Tensor> block_table,
                  const int64_t max_seqlen_q, const int64_t max_seqlen_k,
                  const double dropout_p,
                  const std::optional<double> softmax_scale, const bool causal,
                  const std::vector<int64_t> window_size, const double softcap,
                  const bool deterministic, const bool return_attn_probs,
                  Tensor out, std::optional<Tensor> softmax_lse,
                  std::optional<Tensor> s_dmask) const override {
    if (alibi_slopes || dropout_p != 0.0 || softcap != 0.0 || deterministic ||
        return_attn_probs || softmax_lse || s_dmask ||
        window_size != std::vector<int64_t>{-1, -1}) {
      throw std::invalid_argument(
          "kunlun FlashAttnVarlenFunc received an unsupported option");
    }

    const int64_t batch = cu_seqlens_q.size(0) - 1;
    const int64_t head_num = q.size(1);
    const int64_t head_dim = q.size(2);
    const int64_t kv_head_num = k.size(-2);
    const float scale = static_cast<float>(
        softmax_scale.value_or(1.0 / std::sqrt(static_cast<double>(head_dim))));

    baidu::xpu::api::Context* ctx = VendorContext();

    int rc;
    if (q.dtype() == DataType::kBFloat16) {
      rc = Run<bfloat16>(ctx, stream_, q, k, v, cu_seqlens_q, cu_seqlens_k,
                         block_table, out, batch, max_seqlen_q, max_seqlen_k,
                         head_num, kv_head_num, head_dim, scale, causal);
    } else {
      rc = Run<float16>(ctx, stream_, q, k, v, cu_seqlens_q, cu_seqlens_k,
                        block_table, out, batch, max_seqlen_q, max_seqlen_k,
                        head_num, kv_head_num, head_dim, scale, causal);
    }
    if (rc != 0) {
      throw std::runtime_error(
          "kunlun flash_attention_context_vllm failed, rc=" +
          std::to_string(rc));
    }
  }

 private:
  template <typename T>
  static int Run(baidu::xpu::api::Context* ctx, void* stream, const Tensor& q,
                 const Tensor& k, const Tensor& v, const Tensor& cu_seqlens_q,
                 const Tensor& cu_seqlens_k,
                 const std::optional<Tensor>& block_table, const Tensor& out,
                 int64_t batch, int64_t max_seq_q, int64_t max_seq_kv,
                 int64_t head_num, int64_t kv_head_num, int64_t head_dim,
                 float scale, bool causal) {
    // The vendor kernel reads packed BLHD: the query must be dense
    // ([total_tokens, heads, dim] with unit strides beyond the leading
    // axis). InfiniLM hands over a split view of the fused-QKV GEMM output
    // whose sequence stride spans Q+K+V, so compact it first.
    thread_local std::vector<T> q_dense;
    const T* qp = static_cast<const T*>(q.data());
    if (q.stride(0) != q.size(1) * q.size(2) ||
        q.stride(1) != static_cast<int64_t>(q.size(2)) || q.stride(2) != 1) {
      const int64_t tokens = q.size(0);
      const int64_t heads = q.size(1);
      const int64_t dim = q.size(2);
      // The row copy below assumes a unit inner stride; the base contract
      // (`q.stride(-1) == 1`) guarantees it, but fail loudly if a direct
      // caller ever violates it.
      if (q.stride(2) != 1) {
        return -1005;
      }
      q_dense.resize(tokens * heads * dim);
      xpu_wait(stream);
      for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t h = 0; h < heads; ++h) {
          if (xpu_memcpy(q_dense.data() + (t * heads + h) * dim,
                         static_cast<const char*>(q.data()) +
                             (t * q.stride(0) + h * q.stride(1)) * sizeof(T),
                         dim * sizeof(T), XPU_DEVICE_TO_HOST) != 0) {
            return -1003;
          }
        }
      }
      // Push the compacted copy back to a device buffer.
      thread_local T* q_dense_dev = nullptr;
      thread_local int64_t q_dense_cap = 0;
      if (q_dense_cap < tokens * heads * dim) {
        if (q_dense_dev) xpu_free(q_dense_dev);
        q_dense_dev = nullptr;
        q_dense_cap = 0;
        void* raw = nullptr;
        if (xpu_malloc(&raw, tokens * heads * dim * sizeof(T)) != 0) {
          return -1002;
        }
        q_dense_dev = static_cast<T*>(raw);
        q_dense_cap = tokens * heads * dim;
      }
      if (xpu_memcpy(q_dense_dev, q_dense.data(),
                     tokens * heads * dim * sizeof(T),
                     XPU_HOST_TO_DEVICE) != 0) {
        return -1004;
      }
      qp = q_dense_dev;
    }
    auto* op = static_cast<T*>(const_cast<void*>(out.data()));
    // Prefix-sum lod vectors over the int32 cu_seqlens tensors. `qlod`
    // carries per-sequence query offsets; `kvlod` carries per-sequence
    // TOTAL KV offsets (cached prefix + current chunk), which is what the
    // causal mask tail-aligns queries against. Both copies (host for launch
    // planning, device for in-kernel reads) are materialized.
    thread_local std::vector<int> qlod_host;
    thread_local std::vector<int> kvlod_host;
    qlod_host.assign(batch + 1, 0);
    kvlod_host.assign(batch + 1, 0);
    const int* qlod_dev = static_cast<const int*>(cu_seqlens_q.data());
    const int* kvlod_dev = static_cast<const int*>(cu_seqlens_k.data());
    xpu_wait(stream);
    if (xpu_memcpy(qlod_host.data(), qlod_dev, qlod_host.size() * sizeof(int),
                   XPU_DEVICE_TO_HOST) != 0) {
      return -1000;
    }
    if (xpu_memcpy(kvlod_host.data(), kvlod_dev,
                   kvlod_host.size() * sizeof(int), XPU_DEVICE_TO_HOST) != 0) {
      return -1001;
    }
    baidu::xpu::api::VectorParam<int> qlod{qlod_host.data(), batch + 1,
                                           const_cast<int*>(qlod_dev)};
    baidu::xpu::api::VectorParam<int> kvlod{kvlod_host.data(), batch + 1,
                                            const_cast<int*>(kvlod_dev)};

    // Supports() admits only the paged cache-only mode: K and V are the 4D
    // block-tabled caches (the current chunk was scattered in by
    // `ReshapeAndCacheFlash` just before this call), the kernel reads all KV
    // through the block table, and the query chunk — the tail of each
    // sequence — attends against it. `prefill_len > 0` selects this mode;
    // the dense K/V pointers stay null. The block-table width bounds the
    // per-sequence block count.
    const int64_t block_size = static_cast<int64_t>(k.size(1));
    const int64_t max_blocks_per_seq =
        static_cast<int64_t>(block_table->size(1));
    const int* block_tables_ptr = static_cast<const int*>(block_table->data());
    const auto* k_cache = static_cast<const T*>(k.data());
    const auto* v_cache = static_cast<const T*>(v.data());

    // `alpha` scales the QK^T scores: the kernel applies
    // `(alpha > 0 ? alpha : 1) / sqrt(head_dim)`, so the softmax scale
    // maps back through a multiplication by `sqrt(head_dim)`.
    const float alpha =
        static_cast<float>(scale * std::sqrt(static_cast<double>(head_dim)));

    // Order the vendor kernel against the caller's InfiniRT stream.
    ctx->set_stream(stream);

    const int rc = baidu::xpu::xfa::flash_attention_context_vllm<
        T, T, T, typename CtxVllmBias<T>::type, T,
        typename CtxVllmGemm<T>::type, typename CtxVllmGemm<T>::type, float,
        int, /*quant_mode=*/0>(
        ctx, qp, /*key=*/nullptr, /*value=*/nullptr, /*mask_bias=*/nullptr, op,
        static_cast<std::int64_t>(batch), max_seq_q, max_seq_kv,
        static_cast<std::int64_t>(head_num),
        static_cast<std::int64_t>(head_dim),
        static_cast<std::int64_t>(kv_head_num), /*alibi_slopes=*/nullptr, qlod,
        kvlod, /*mask_bias_shape=*/{}, /*alibi_slopes_shape=*/{}, causal,
        /*is_qkv_fusion=*/false,
        /*qkv_layout=*/0x10 /*ATTN_BLHD*/, /*vsl_type=*/0x10 /*ATTN_LVSL*/,
        /*q_maxptr=*/nullptr,
        /*k_maxptr=*/nullptr, /*v_maxptr=*/nullptr, /*o_maxptr=*/nullptr,
        /*vo_head_dim=*/-1, k_cache, v_cache,
        /*k_cache_maxptr=*/nullptr, /*v_cache_maxptr=*/nullptr, block_size,
        max_blocks_per_seq,
        /*prefill_len=*/max_seq_q, block_tables_ptr, alpha,
        /*swa_left_size=*/-1, /*swa_right_size=*/-1, /*softmax_lse=*/nullptr,
        /*unpadded_lse=*/false);
    return rc;
  }

  static baidu::xpu::api::Context* VendorContext() {
    thread_local baidu::xpu::api::Context* ctx = nullptr;
    if (ctx == nullptr) {
      ctx = baidu::xpu::api::create_context();
      // The context is reused across calls; its stream is bound per call
      // (`ctx->set_stream(stream)` before each vendor launch) so vendor
      // kernels order against the caller's InfiniRT submissions.
    }
    return ctx;
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_KUNLUN_FLASH_ATTN_VARLEN_FUNC_KERNEL_H_
