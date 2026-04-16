# Fused attention / dispatch-count design note

## Executive summary

Reading vllm-ascend before designing revealed that the original "fused attention
block" hypothesis was wrong. **Ascend does not fuse attention + rope +
reshape_and_cache into a single custom op.** Both plugins go through the same
vLLM-core `torch.ops.vllm.unified_attention_with_output` wrapper 2304 times on a
64-forward, 36-layer run (Qwen2.5-3B), exactly matching one attn call per layer
per forward.

The real lever is different: **ascend pays ~6,976 Python-level `_ops.__call__`
dispatches per 64-forward run, infini pays 30,080** (4.31x). The gap is in the
*non-attention* per-layer ops (gemm, add_rms_norm, rope, swiglu). Ascend keeps
those out of the `torch.ops.vllm.*` dispatch path on replay; infini pays two
`_ops.__call__` hops per op (outer `torch.ops.vllm.infini_<op>` -> inner
wrapper function -> `infini.ops.<kernel>`).

Break-down of dispatch counts per 64-forward run (Qwen2.5-3B, graph mode,
decode steady state):

| Op | infini ncalls | ascend ncalls | Who pays more |
| --- | ---: | ---: | --- |
| `unified_attention_with_output` | 2,304 | 2,304 | parity |
| `npu_fused_infer_attention_score` (inside FIA) | — | 2,304 | ascend only (inside attention) |
| `atb._npu_reshape_and_cache` | — | 2,304 | ascend only (inside attention) |
| `vllm.infini_unquantized_gemm` | **18,496** | — | infini only |
| `vllm.unquantized_gemm` | — | 64 | ascend only (LM head, 1/forward) |
| `vllm.infini_add_rms_norm` | **4,608** | — | infini only |
| `vllm.infini_rotary_embedding_v2` | **2,304** | — | infini only |
| `vllm.infini_swiglu` | **2,304** | — | infini only |
| **Total `_ops.__call__`** | **30,080** | **6,976** | infini pays 4.3x |

## Root cause

Comparing call counts against layers × forwards:

- 18,496 gemm dispatches / 64 forwards = **289 per forward**. Qwen2.5-3B has 36 layers × (QKV, out_proj, up_proj, gate_proj, down_proj) = 36 × 7 = 252, plus ~36 additional sampling / LM head ops. Close to 289 — confirms one `torch.ops.vllm.infini_unquantized_gemm` dispatch **per gemm**, **per layer**, **per forward**.
- 4,608 add_rms_norm / 64 = 72 per forward = 36 × 2 (input + post-attn). **One dispatch per add_rms_norm call.**
- Ascend's 64 `unquantized_gemm` calls / 64 = **1 per forward**. That's the LM head only.

**Inference**: ascend's compiled FX graph replaces the per-layer `torch.ops.vllm.unquantized_gemm` node with a direct kernel call (or a torch-native op that doesn't go through `torch.ops._ops.__call__` instrumentation). Infini's compiled FX graph re-invokes the custom op wrapper on every replay.

This is actually the **vLLM v1 PIECEWISE semantics**. The FX graph between piecewise attention boundaries contains the custom-op nodes; when replayed, each call is a full Python dispatch. Ascend has some mechanism to short-circuit this — either by registering their custom ops without the `vllm::` namespace prefix, or by using `use_direct_call=True` somewhere, or by compiling the graph differently. Still investigating.

## Options for closing the gap

### Option F1 — Short-circuit `infini_unquantized_gemm` (biggest lever)

18,496 gemm dispatches × ~109 us cProfile-measured per-call = **2.0 s** cumulative time (28% of infini's 7.06 s cProfile wall).

The wrapper structure today:

```python
# ops/linear.py
def infini_unquantized_gemm(layer, x, weight, bias):
    out_shape = (*x.shape[:-1], weight.shape[0])
    x_2d = x.view(-1, x.shape[-1]) if x.dim() > 2 else x
    out = torch.ops.vllm.infini_unquantized_gemm(x_2d, weight, bias)  # dispatch #1
    return out.view(out_shape)

# The dispatcher routes to `_infini_gemm`:
def _infini_gemm(x, weight, bias=None):                                # dispatch #2
    stream = current_stream_ptr()
    out = torch.empty(...)
    infini.ops.linear(x, weight, bias, ..., out=out, stream=stream)
    return out
```

Two `torch._ops._ops.__call__` hops per gemm on eager. Under Dynamo, the graph traces the outer `torch.ops.vllm.infini_unquantized_gemm` call, so replay also goes through the outer dispatch.

**Proposed**: replace the `torch.ops.vllm.infini_unquantized_gemm` wrapper with a direct `infini.ops.linear` call in `infini_unquantized_gemm()`. Keep the `direct_register_custom_op` registration so the op is still addressable from `torch.ops.vllm.*` (required for Dynamo fake tensors), but when called from eager Python, skip the dispatch — call `_infini_gemm` directly. Pseudocode:

```python
def infini_unquantized_gemm(layer, x, weight, bias):
    out_shape = (*x.shape[:-1], weight.shape[0])
    x_2d = x.view(-1, x.shape[-1]) if x.dim() > 2 else x
    # Direct call: skip torch.ops.vllm.* dispatch, infini.ops.linear is
    # already the underlying kernel.
    out = _infini_gemm(x_2d, weight, bias)
    return out.view(out_shape)
```

**Risk**: Dynamo may no longer see the call as a custom-op node in the FX graph. If the graph is PIECEWISE-compiled, each linear is a separate graph node today; if we bypass the custom-op path, Dynamo might inline the `_infini_gemm` body and its `current_stream_ptr()` / `torch.empty()` into the graph, which could mis-capture the stream or the output buffer. **Must test Dynamo tracing works after the change.**

**Expected savings**: if we cut half the gemm dispatches (one per-call instead of two), save ~1.0 s of cProfile host time (14% of wall). If we save all gemm-side overhead, closer to 2.0 s.

### Option F2 — Same for add_rms_norm, rope, swiglu

Same pattern repeats in `ops/layernorm.py`, `ops/rotary_embedding.py`, `ops/activation.py`. Each does `torch.ops.vllm.infini_<op>(...)` in the outer wrapper and routes to `_infini_<op>(...)` in the inner.

Combined savings (cProfile):
- `infini_add_rms_norm`: 4608 calls × ~205 us = 0.94 s
- `infini_rotary_embedding_v2`: 2304 × ~299 us = 0.69 s
- `infini_swiglu`: 2304 × ~184 us = 0.42 s

If F1+F2 halve dispatch overhead for all four op families, savings ~**2.0 s** (28% of wall). Combined with the already-shipped stream cache, would move infini host time from 7.06 s → ~5.0 s, projected +20-30% throughput on 3B graph.

### Option F3 — Fuse rope + attention + reshape_and_cache into ONE `vllm.*` op

This was the original proposal. Now known to **not** match what ascend does. Ascend does the three inside `InfiniAttentionImpl.forward` which is called via `unified_attention_with_output` — already in one dispatch. Our `InfiniAttentionImpl.forward` calls `infini.ops.*` kernels directly (not via `torch.ops.vllm.*`), so there are no extra dispatches to fuse. This option is a no-op — skip.

## Recommended plan

1. **F1 first**: patch `ops/linear.py` to call `_infini_gemm` directly in the OOT entrypoint, bypass `torch.ops.vllm.*` on the eager call path. Keep the custom-op registration for Dynamo fake tensors.
2. Gate on `INFINI_FUSED_ATTN=1` (even though it's not actually attention — rename env var or document alias).
3. Token-level diff on both 0.5B and 3B.
4. If Dynamo tracing breaks, roll back and reconsider.
5. If F1 cleanly works, proceed to F2 for the other three op families.
6. Re-measure both `vllm bench throughput` and cProfile to confirm the dispatch count drops.
7. Record in `docs/perf/e2e_progress.md`.

## Kill-switch

`INFINI_DIRECT_OPS=0` (default on) restores the `torch.ops.vllm.*` dispatch path. Set to 0 to bisect if correctness breaks.

## Unknowns I couldn't answer from reading

1. **Why does ascend's `unquantized_gemm` only show 64 calls?** They use the same `direct_register_custom_op` pattern but their dispatch count is 289x lower. Either their compiled FX graph has the gemm inlined as a torch-native op, or there's a CompilationConfig flag (like `fullgraph` or `custom_ops` whitelist) that excludes `unquantized_gemm` from the piecewise graph. Need to diff the compiled `<eval_with_key>.*:forward` FX code between the two.

2. **Does `direct_register_custom_op` route through `_ops.__call__` at replay?** Or does Dynamo's FX graph call the underlying function directly? If the latter, F1 might not actually save dispatches (the replay would be fast either way, and only the first tracing pass is slow). Need to verify with a quick microbench.

3. Whether Dynamo's FX tracing requires the `torch.ops.vllm.*` hop to preserve the custom-op boundary, or if direct calls to `_infini_gemm` still trace cleanly into the graph.

I'll validate (2) and (3) with small tests before implementing F1.

## Request for review

Pings for team-lead:
- OK with the F1→F2 sequence?
- OK to drop F3 from the plan?
- OK with the `INFINI_DIRECT_OPS` kill-switch (name subject to bikeshed)?
- Do you want me to first answer the "unknown #1" (ascend's actual dispatch-count reduction mechanism) before F1, or land F1 and cross-check against ascend after?
