# FlashAttention via FlashInfer Integration

## Problem

FlashAttention is the only operator in InfiniOps without an NVIDIA
implementation. FlashInfer provides a header-only C++ API with
state-of-the-art attention kernels for both prefill and decode.

## Solution

Integrate FlashInfer as a header-only dependency. Wrap its C++ API in
InfiniOps's `CudaFlashAttention` operator class, mapping InfiniOps's
`FlashAttention` base class parameters to FlashInfer's param structs.

---

## Integration approach

1. Add FlashInfer headers to `third_party/flashinfer/include/`.
2. Add FlashInfer's CUTLASS dependency to `third_party/flashinfer/3rdparty/cutlass/`.
3. Update `src/CMakeLists.txt` to add include paths when `WITH_NVIDIA=ON`.
4. Create `src/cuda/flash_attention/kernel.h` wrapping FlashInfer's
   `SinglePrefillWithKVCacheDispatched`.
5. Create `src/nvidia/flash_attention/kernel.h` as the nvidia wrapper.

## Parameter mapping

| InfiniOps | FlashInfer |
|-----------|-----------|
| `query [T, N, D]` | `params.q`, `params.qo_len=T` |
| `key [S, Nkv, D]` | `params.k`, `params.kv_len=S` |
| `value [S, Nkv, D]` | `params.v` |
| `num_heads` | `params.num_qo_heads` |
| `num_kv_heads` | `params.num_kv_heads` |
| `head_size` | template `HEAD_DIM` + `params.head_dim` |
| `scale` | `params.sm_scale` |
| `causal` | `MaskMode::kCausal` vs `MaskMode::kNone` |
| `window_left` | `params.window_left` |
| `output [T, N, D]` | `params.o` |

## Scope

Initial implementation covers **single-request prefill** (non-paged,
contiguous KV). This handles the standard attention pattern. Paged KV
cache and batch decode can be added later.

## Head dimension dispatch

FlashInfer requires HEAD_DIM as a compile-time template parameter.
Dispatch at runtime:

```cpp
switch (head_size) {
  case 64:  return launch<64>(...);
  case 128: return launch<128>(...);
  case 256: return launch<256>(...);
  default:  assert(false && "unsupported head_size");
}
```

## Data type dispatch

Use InfiniOps's existing `DispatchFunc` for dtype → (half, nv_bfloat16,
float) mapping.

## Files

| File | Action |
|------|--------|
| `third_party/flashinfer/` | New: FlashInfer headers (git submodule) |
| `src/CMakeLists.txt` | Modify: add FlashInfer include path |
| `src/cuda/flash_attention/kernel.h` | New: CudaFlashAttention wrapper |
| `src/nvidia/flash_attention/kernel.h` | New: nvidia specialization |
| `tests/test_flash_attention.py` | Modify: enable CUDA tests |
