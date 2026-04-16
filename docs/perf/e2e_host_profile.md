# Host-side Python profile — vllm-infini vs vllm-ascend

Date: 2026-04-17.
Workload: Qwen2.5-3B-Instruct, 64 prompts × 64 output tokens (after 8×8 warmup), PIECEWISE graph mode, fp16, 1 NPU (device 1), CANN 8.5.1.
Method: `cProfile.Profile()` around the profiled `llm.generate()` call only; warmup excluded. `VLLM_ENABLE_V1_MULTIPROCESSING=0` to keep the engine in-process so `cProfile` captures everything.

Harness: `/tmp/cprofile_runner.py`. Diff tool: `/tmp/cprof_compare.py`.

## Headline

| Metric | infini | ascend | Ratio |
| --- | ---: | ---: | ---: |
| Total wall time (cProfile) | **7.056 s** | **2.785 s** | **2.53x** |
| Per-`step()` time | 108 ms | 42 ms | 2.57x |
| `_model_forward` cumtime | 6.295 s | 1.959 s | **3.21x** |
| `torch._ops._ops.__call__` ncalls | 30,080 | 6,976 | **4.31x** |
| `nn.Module._wrapped_call_impl` ncalls | 4,992 | 2,661 | 1.88x |
| `piecewise_backend.__call__` ncalls | 2,368 | 37 | 64x |

Note: cProfile adds a large constant overhead (~9× slowdown). Absolute times are cProfile-inflated, but relative comparisons hold.

## Root cause of the 2.53x host-time gap

The gap is **Python op-dispatch overhead**. Infini exposes every layer op (linear, norm, rope, swiglu, attention, reshape_and_cache) as an individual `torch.ops.vllm.*` custom op, each wrapped in a Python function that calls `infini.ops.*`. vllm-ascend collapses the per-layer work into fewer, larger custom ops (notably `unified_attention_with_output`) so Python dispatch fires far less often.

### Top infini-only host costs (functions absent from ascend profile)

| Function | ncalls | cumtime (s) | per-call (us) |
| --- | ---: | ---: | ---: |
| `_stream.py:current_stream_ptr` | **20,864** | **1.955** | 94 |
| `ops/linear.py:_infini_gemm` | 9,280 | 1.666 | 179 |
| `attention/backend.py:forward` | 2,304 | 1.272 | 552 |
| `torch._ops.vllm.infini_unquantized_gemm` | 18,496 | 2.025 | 109 |
| `torch._ops.vllm.infini_add_rms_norm` | 4,608 | 0.944 | 205 |
| `ops/layernorm.py:_infini_add_rms_norm` | 4,608 | 0.860 | 187 |
| `torch._ops.vllm.infini_rotary_embedding_v2` | 2,304 | 0.690 | 299 |
| `ops/rotary_embedding.py:_infini_rotary_embedding_v2` | 2,304 | 0.642 | 279 |
| `infini.ops.linear` (C++ binding) | 9,280 | 0.546 | 59 |
| `infini.ops.paged_attention` (C++ binding) | 2,268 | 0.517 | 228 |
| `torch._ops.vllm.infini_swiglu` | 2,304 | 0.424 | 184 |
| `ops/activation.py:_infini_swiglu` | 2,304 | 0.393 | 171 |
| `infini.ops.add_rms_norm` (C++ binding) | 4,608 | 0.332 | 72 |
| `infini.ops.apply_rotary_pos_emb` (C++ binding) | 2,304 | 0.265 | 115 |
| `torch.empty` | 13,890 | 0.261 | 19 |
| `infini.ops.reshape_and_cache` (C++ binding) | 2,304 | 0.260 | 113 |

**Notes on shape**: 64 forwards × 36 layers × *n* ops/layer = call counts.
- 2,304 = 64 × 36 (per-layer hot ops: rope, swiglu, attention, reshape_and_cache).
- 4,608 = 64 × 36 × 2 (add_rms_norm pair per layer: input+post-attn).
- 9,280 = 64 × 145 ≈ 36 × 4 + ~1 LM head; real count is 64 × (36 × 4 MLP/attn projections + ~1) ≈ 9,280 direct `_infini_gemm` calls.

### Top ascend-only host costs (functions absent from infini profile)

| Function | ncalls | cumtime (s) |
| --- | ---: | ---: |
| `model_runner_v1.py:_model_forward` | 64 | 1.959 |
| `attention_v1.py:forward` | 2,304 | 0.982 |
| `acl_graph.py:__call__` | 2,368 | 0.710 |
| `attention_v1.py:forward_impl` | 2,304 | 0.661 |
| `attention_v1.py:forward_fused_infer_attention` | 2,304 | 0.606 |
| `torch._ops.npu.npu_fused_infer_attention_score` | 2,304 | 0.359 |
| `attention_v1.py:reshape_and_cache` | 2,304 | 0.228 |
| `torch_npu._C.replay` | 2,331 | 0.185 |
| `torch._ops.atb._npu_reshape_and_cache` | 2,304 | 0.147 |

Their attention wrapper (`attention_v1.py:forward`) takes 425 us/call — **1.3× faster than ours at 552 us/call** — and it absorbs RoPE + flash-attention + cache_update + a `reshape_and_cache` downcall in one wrapper. They also pay `graphs.py:replay` / `_C.replay` per graph segment, but the total is only 0.19 s.

## Top-3 Python deltas, ranked by fix leverage

### #1 — `current_stream_ptr` is called ~326× per forward for **94 us each** (1.955 s total)

Every `infini.ops.*` call at `ops/linear.py:26`, `ops/layernorm.py:22,36`, `ops/activation.py:17`, `ops/rotary_embedding.py`, `attention/backend.py` calls `current_stream_ptr()`. The implementation is:

```python
def current_stream_ptr() -> int:
    stream = torch.cuda.current_stream()
    return getattr(stream, "npu_stream", None) or stream.cuda_stream
```

On each call: Python dispatch → `torch.cuda.current_stream()` (patched to `torch.npu.current_stream()`) → Python property getter → `getattr` fall-through → int return. ~94 us per hit × 20,864 hits = **1.955 s / 7.056 s = 27.7% of wall time**.

**Proposed fix** (minimal, `_stream.py` scope): cache the stream handle at the start of each forward pass. Stream switches across a forward are rare (and when they happen, e.g. sampler side-stream, they pass the stream explicitly). Two candidate implementations:

A. Expose a `forward_local_stream(ctx)` context manager that resolves the pointer once and stashes it in a thread-local `ctx`; ops read from the local.

B. Add a module-level `_cached_stream_ptr` that is invalidated via vLLM's `set_forward_context`. Simpler; matches how metadata is cached today.

Expected savings: ~1.9 s of cProfile overhead → infini/ascend wall-time ratio drops from 2.53× to ~1.83×. Converted to real tok/s using the graph-mode baseline (infini 5,299 / ascend 10,148, ratio 52.2%), this should recover ~27% throughput → **roughly 73-75% of vllm-ascend** in graph mode. Back-of-envelope only; needs measurement.

### #2 — `torch._ops._ops.__call__` fires 4.31× as often as in ascend

30,080 vs 6,976. Every `infini_<op>` is dispatched as `torch.ops.vllm.<op>()`, which in turn calls the Python wrapper, which calls `infini.ops.<op>()`. That's two `_ops.__call__` per "logical" op. Meanwhile vllm-ascend's `unified_attention_with_output` collapses attention + rope + cache into one dispatch.

**Proposed fix**: bigger change. Register a single `torch.ops.vllm.infini_attention_block` that takes `(qkv, kv_cache, metadata, ...)` and performs all of rope + flash_attention/paged_attention + reshape_and_cache inside the wrapper. Eliminates ~3 dispatches per layer × 36 layers × 64 steps = 6,912 `_ops.__call__` calls.

This is structurally bigger (touches `attention/backend.py`, `ops/rotary_embedding.py`, and needs a new `direct_register_custom_op`). Worth tackling *after* #1 to measure the isolated impact.

### #3 — `torch.empty` fires 13,890× (0.261 s) — infini-only

Likely per-op output allocation. vllm-ascend doesn't show this; their kernels reuse caller-provided output buffers. Not load-bearing on its own (~3.7% of host time) but compounds with #2 — if we fuse the attention block we can share buffers.

## Recommendations in priority order

1. **Do #1 first.** Smallest code change (stream cache in `_stream.py`), biggest win (~27% of host wall time). No operator-side coordination needed.
2. Re-measure after #1. If we hit 75%+, evaluate whether #2 is still worth it.
3. If still below target, scope #2 (fused attention block) as a medium-sized patch to `vllm-infini/attention/backend.py`. Requires no `src/ascend/` changes — it's a pure plugin-side fusion of existing `infini.ops.*` calls.
4. Skip #3 in isolation; tackle it as a side effect of #2.

## Raw .pstats

- `/tmp/cprof_infini_3b_graph.pstats`
- `/tmp/cprof_ascend_3b_graph.pstats`
- `/tmp/cprof_infini_0p5b_graph.pstats`
- `/tmp/cprof_ascend_0p5b_graph.pstats`
- `/tmp/cprof_infini_0p5b_graph_cached.pstats` (with stream cache prototype — see below)

Both files are inside container `infiniops-bench-ascend-v2` (mounted at `/tmp`). The harness that produced them is `/tmp/cprofile_runner.py`; the diff tool is `/tmp/cprof_compare.py`.

## 0.5B canary confirms the same signal

Same analysis on Qwen2.5-0.5B (expected to amplify host-side overhead on small kernels):

| Metric | infini | ascend | Ratio |
| --- | ---: | ---: | ---: |
| Total wall time (cProfile) | 4.908 s | 1.907 s | 2.57x |
| `_ops.__call__` ncalls | 20,096 | 4,672 | 4.30x |
| `current_stream_ptr` ncalls / cumtime | 13,952 / 1.317 s | N/A | 26.8% of host wall |

0.5B ratio (2.57x) is basically identical to 3B (2.53x), confirming the per-op Python-dispatch overhead dominates and the fix target is robust across model sizes.

## Stream-cache prototype — correctness REGRESSION, reverted

Prototype: cache the resolved pointer in `_stream.py`, invalidate via a wrapper around `torch.npu.set_stream` in `_patches.py`. Gated behind `INFINI_CACHE_STREAM` env var (default on).

cProfile impact: 0.5B host wall dropped from 4.908 s → 3.568 s (**-27.3%**, matches prediction).

**Throughput impact** (measured with full `vllm bench throughput`, 256 prompts, 128/128):

| Model | Mode | Before | After cache | Ratio | Delta |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5B | graph | 7,940 tok/s | **9,624 tok/s** | 62.0% of ascend | +21.2% |
| 3B   | graph | 5,299 tok/s | **6,091 tok/s** | 60.0% of ascend | +15.0% |
| 3B   | eager | 5,291 tok/s | **5,835 tok/s** | 87.2% of ascend | **+10.3%, passes 80% target** |

**BUT**: `/tmp/correctness_check.py` with `VLLM_PLUGINS=infini` on Qwen2.5-3B-Instruct fails:

| Config | Token-match vs vllm-ascend |
| --- | --- |
| baseline (no cache)            | 6/6 exact match |
| with cache, MP=1 (default)     | 5/6 — prompt 0 produces `!!!!!…` x64 (all `token_id=0`) |
| with cache, MP=0               | **0/6 — all prompts produce garbage** |

The throughput benches use `ignore_eos=True` and don't verify outputs, which is why they didn't flag the regression. Only the correctness diff script caught it.

**Root cause** (hypothesised, not yet verified): some code path switches streams without going through `torch.npu.set_stream`. Candidates not yet ruled out:

- `torch_npu._C._npu_setStream` called directly, bypassing the Python wrapper.
- A `StreamContext.__enter__/__exit__` path that uses a different entry point.
- A graph-capture / compile hook that briefly switches streams during a dummy forward.
- `forward_context.set_forward_context` establishing a stream via a different mechanism.

The MP=0 case (0/6 broken) is worse than MP=1 (5/6) — likely because MP=0 runs more init/warmup in the same process, filling the cache with a "wrong" pointer earlier in the lifecycle.

**Revert state**: both `_stream.py` and `_patches.py` are back to clean (no cache committed).

## Next steps for the stream-cache lever

Do NOT land the `_stream.py`-level cache without first nailing down the bug. Safer designs to investigate:

1. **Bracket-style cache per forward**: the model-runner explicitly calls `_stream.begin_forward()` at the top of `execute_model` and `_stream.end_forward()` after, which set/clear the cache. No reliance on `set_stream` hooks. Needs a tiny hook in the model-runner's execute path (pluggable via `_patches.py`).
2. **Invalidate on every `set_forward_context` / forward-end boundary**: wrap `vllm.forward_context.set_forward_context` (a contextmanager) so entering/exiting a forward pass invalidates the cache. Keeps `_stream.py` standalone. Probably the safest and simplest option.

Option 2 expected savings: still ~1.8 s of 7.1 s = ~25% host time (one resolve and reuse per forward, not per op). Essentially the same win as the naive cache, but correctness-safe because each forward starts fresh.

Pending team-lead decision on whether to invest in option 2 or pivot to a different lever (e.g., the fused attention block that eliminates several per-layer dispatches and side-steps the `current_stream_ptr` question).
