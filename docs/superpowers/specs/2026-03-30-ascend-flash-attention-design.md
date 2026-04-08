# Ascend Flash Attention & Reshape-And-Cache Design

**Date:** 2026-03-30
**Status:** Approved
**Scope:** Two new operators for the Ascend backend, compatible with vLLM input layout conventions.

## Overview

Add `FlashAttention` and `ReshapeAndCache` operators to InfiniOps targeting the Ascend NPU backend. The operators wrap CANN's `aclnnFusedInferAttentionScore` (FIA) API and accept vLLM-compatible TND (token-major) tensor layouts, enabling direct integration with vLLM's attention pipeline.

## Operator 1: FlashAttention

### Interface

```cpp
// src/base/flash_attention.h
class FlashAttention : public Operator<FlashAttention> {
 public:
  FlashAttention(
      const Tensor query,             // [num_tokens, num_heads, head_size] TND
      const Tensor key,               // TND or paged cache [num_blocks, KV_N, block_size, D]
      const Tensor value,
      std::optional<Tensor> block_table,  // [num_reqs, max_blocks_per_req], INT32
      std::optional<Tensor> cu_seqlens_q, // [num_reqs + 1], INT64
      std::optional<Tensor> cu_seqlens_kv,// [num_reqs + 1], INT64
      int64_t num_heads,
      int64_t num_kv_heads,
      int64_t head_size,
      double scale,                    // 1/sqrt(head_size)
      int64_t sparse_mode,            // 3 = causal (right-down triangular)
      int64_t block_size,             // 0 = no paging, else 128/256/384/512
      Tensor output                   // [num_tokens, num_heads, head_size]
  );

  virtual void operator()(
      const Tensor query, const Tensor key, const Tensor value,
      std::optional<Tensor> block_table,
      std::optional<Tensor> cu_seqlens_q,
      std::optional<Tensor> cu_seqlens_kv,
      int64_t num_heads, int64_t num_kv_heads, int64_t head_size,
      double scale, int64_t sparse_mode, int64_t block_size,
      Tensor output
  ) const = 0;
};
```

### Tensor Layout

All tensors use TND (token-major) layout to match vLLM conventions:

| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `query` | `[num_tokens, num_heads, head_size]` | fp16/bf16 | Concatenated query tokens |
| `key` | `[num_tokens, num_kv_heads, head_size]` or `[num_blocks, KV_N, block_size, D]` | fp16/bf16 | Input K or paged cache |
| `value` | Same shape as `key` | fp16/bf16 | Input V or paged cache |
| `output` | `[num_tokens, num_heads, head_size]` | fp16/bf16 | Attention output |
| `block_table` | `[num_reqs, max_blocks_per_req]` | INT32 | Paged KV cache block mapping |
| `cu_seqlens_q` | `[num_reqs + 1]` | INT64 | Cumulative query sequence lengths |
| `cu_seqlens_kv` | `[num_reqs + 1]` | INT64 | Cumulative KV sequence lengths |

### ACLNN FIA Mapping

The Ascend backend (`src/ascend/flash_attention/kernel.h`) wraps `aclnnFusedInferAttentionScore`:

| InfiniOps | ACLNN FIA | Notes |
|-----------|-----------|-------|
| `query [T,N,D]` | `query` as `[1,N,T,D]` BNSD | Reshape (view, no copy) |
| `key` (paged cache) | `key` as `aclTensorList*` | Single-element list pointing to cache |
| `value` (paged cache) | `value` as `aclTensorList*` | Same as key |
| `block_table` | `blockTable` | Direct pass-through |
| `cu_seqlens_q` | `actualSeqLengths` | Extract to host `aclIntArray*` |
| `cu_seqlens_kv` | `actualSeqLengthsKv` | Extract to host `aclIntArray*` |
| `num_heads` | `numHeads` | |
| `num_kv_heads` | `numKeyValueHeads` | Supports GQA |
| `scale` | `scaleValue` | |
| `sparse_mode` | `sparseMode` | 3 = causal |
| `block_size` | `blockSize` | |
| `output [T,N,D]` | `attentionOut` as `[1,N,T,D]` | Reshape back |

**Internal defaults (not exposed):**

- `inputLayout` = `"BNSD"`
- `pseShift` = nullptr (no position encoding shift)
- `attenMask` = nullptr (causal handled by `sparseMode=3`)
- `preTokens` / `nextTokens` = `2147483647` (INT_MAX)
- `innerPrecise` = 0 (high precision mode)
- `softmaxLseFlag` = false
- All quantization parameters = nullptr

### Workflow

1. Reshape TND input tensors to BNSD views (no memory copy)
2. Extract `cu_seqlens_q`/`cu_seqlens_kv` to host-side `aclIntArray*`
3. Build ACL tensor descriptors via `ascend::buildAclTensor()`
4. Create `aclTensorList*` for key/value (single-element list wrapping the cache tensor)
5. Call `aclnnFusedInferAttentionScoreGetWorkspaceSize`
6. Allocate workspace via `WorkspacePool::ensure()`
7. Call `aclnnFusedInferAttentionScore`
8. Destroy all ACL descriptors

### Constraints

- **Dtypes:** float16, bfloat16 only
- **head_size:** must be 16-byte aligned (multiple of 8 for fp16, 4 for bf16), max 512
- **num_heads:** max 256
- **block_size:** 128, 256, 384, or 512 (multiple of 128). 0 disables paging
- **KV cache format:** `(num_blocks, KV_N, block_size, D)` preferred (better performance than `(num_blocks, block_size, H)`)
- **GQA:** `num_heads % num_kv_heads == 0`, ratio <= 64
- **Paged attention requires:** `block_table` present, `cu_seqlens_kv` provided, `block_size >= 128`

## Operator 2: ReshapeAndCache

### Interface

```cpp
// src/base/reshape_and_cache.h
class ReshapeAndCache : public Operator<ReshapeAndCache> {
 public:
  ReshapeAndCache(
      const Tensor key,               // [num_tokens, num_kv_heads, head_size]
      const Tensor value,             // [num_tokens, num_kv_heads, head_size]
      const Tensor kv_cache,          // [num_blocks, block_size, num_kv_heads, head_size]
      const Tensor slot_mapping,      // [num_tokens], INT64
      Tensor kv_cache_out             // same shape as kv_cache (in-place)
  );

  virtual void operator()(
      const Tensor key, const Tensor value,
      const Tensor kv_cache, const Tensor slot_mapping,
      Tensor kv_cache_out
  ) const = 0;
};
```

### Behavior

Scatter-writes new key/value tokens into the paged KV cache. For each token `i`:

```
slot = slot_mapping[i]
block_idx = slot // block_size
offset = slot % block_size
kv_cache_out[block_idx, offset, :, :] = key[i, :, :]
```

### Implementation

Start with `aclrtMemcpy`-based element-wise copy with stride arithmetic (no custom AscendC kernel). Optimize later if profiling shows this is a bottleneck.

## File Structure

```
src/base/flash_attention.h             # Abstract base class
src/base/reshape_and_cache.h           # Abstract base class
src/ascend/flash_attention/kernel.h    # Ascend specialization
src/ascend/reshape_and_cache/kernel.h  # Ascend specialization
tests/test_flash_attention.py          # Operator tests
tests/test_reshape_and_cache.py        # Operator tests
```

## Testing Strategy

### FlashAttention Tests

Tests follow the `Payload` / `auto_act_and_assert` pattern from `conftest.py`:

- **Prefill (no block table):** single sequence, multi-sequence with `cu_seqlens`
- **Decode (with block table):** single token per request with paged KV cache
- **GQA:** `num_kv_heads < num_heads`
- **Causal masking:** `sparse_mode=3`
- **Dtypes:** fp16, bf16 (skipped on Ascend for unsupported dtypes)
- **Reference:** PyTorch `scaled_dot_product_attention` with causal mask

### ReshapeAndCache Tests

- Write single token into empty paged cache, verify correct slot placement
- Write batch of tokens with contiguous slot mapping
- Write batch with non-contiguous slot mapping (holes in cache)
- **Reference:** manual scatter via NumPy indexing

### Device Filtering

Tests use `device="npu"` parametrization. Use `-k "not cpu"` to select Ascend tests (avoids substring match with "input").

## Python Bindings

Auto-generated by `scripts/generate_wrappers.py`. Usage:

```python
import infini

# Free function
out = infini.ops.flash_attention(
    query, key, value,
    block_table=block_table,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_kv=cu_seqlens_kv,
    num_heads=32, num_kv_heads=8, head_size=128,
    scale=1.0/128**0.5, sparse_mode=3, block_size=128,
    output=out
)

# ReshapeAndCache
infini.ops.reshape_and_cache(key, value, kv_cache, slot_mapping, kv_cache)
```

## Decisions Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ACLNN API | `aclnnFusedInferAttentionScore` (FIA) | Single API for prefill + decode, matches vllm-ascend's primary path |
| Tensor layout | Accept TND, reshape to BNSD internally | Matches vLLM conventions, simpler Python adapter |
| Operator scope | FlashAttention + ReshapeAndCache | Covers full vLLM attention pipeline: cache write + attention computation |
| Quantization | Not exposed in initial version | YAGNI — can add quantization params later |
| ReshapeAndCache impl | `aclrtMemcpy` with strides | Simplest, no custom kernel. Optimize after profiling. |
| KV cache format | `(num_blocks, KV_N, block_size, D)` | Better performance per ACLNN docs |

## Out of Scope

- MLA (Multi-head Latent Attention) support
- Quantized attention (INT8 input/output)
- Custom AscendC kernels for hot-path optimization
- Full vLLM `AttentionBackend` implementation
- Speculative decoding support
- Sparse Flash Attention (DSA)
