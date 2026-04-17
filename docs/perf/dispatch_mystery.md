# Dispatch-count mystery — resolved

**Question**: why does vllm-ascend make ~6,976 Python-level `torch._ops._ops.__call__` dispatches vs vllm-infini's ~30,080 on the same 64-forward Qwen2.5-3B graph-mode run, when both plugins use visually-identical `direct_register_custom_op` signatures?

**Answer**: vllm-ascend installs a full **FX graph fusion pass manager** plus a **custom inductor-based compile backend** that collapses per-layer op chains into fewer fused ops. vllm-infini uses a pass-through compiler that returns the FX graph unchanged, so our per-layer custom-op nodes stay in the graph and re-dispatch on every replay.

## Evidence

### Team-lead's cProfile-artefact hypothesis — falsified

`torch._ops.atb._npu_reshape_and_cache` (pure C++ op, TORCH_LIBRARY-registered)
shows up in cProfile with `ncalls=2304` in the ascend run. So C++ ops DO get counted by cProfile. The mystery isn't "cProfile misses C++ inlined calls".

### Signature diff — identical, ruled out

All `direct_register_custom_op` call sites use the same signature on both sides:

```python
direct_register_custom_op(
    op_name="<name>",
    op_func=<func>,
    fake_impl=<fake>,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
```

Checked: `vllm-infini/ops/{linear.py,layernorm.py,rotary_embedding.py,activation.py}`, `vllm-ascend/ops/mla.py`, `vllm-ascend/ops/register_custom_ops.py`, `vllm-ascend/patch/worker/patch_unquantized_gemm.py`. No tag/flag differences.

### Callee-count from `<eval_with_key>.*:forward`

Per FX-graph frame (pstats caller-callee analysis):

| Caller | Plugin | Dominant callee type | ncalls per frame |
| --- | --- | --- | ---: |
| `<eval_with_key>.50:forward` | infini | `_ops.py:__call__` (many entries) | **~8 dispatches per piece** |
| `<eval_with_key>.58:forward` | ascend | `_ops.py:__call__` (many entries) | **~1 dispatch per piece** |

Same 64 forwards, same 36 layers, same piecewise-graph topology (2368 piece invocations on both). But each infini piece contains ~8 per-op custom-op dispatches in its body; each ascend piece contains ~1. **The FX graphs are structurally different**.

### Configuration diff

| Setting | infini | ascend |
| --- | --- | --- |
| `CompilationConfig.backend` | `""` (empty) | set via `platform.get_compile_backend()` |
| `platform.get_compile_backend()` | N/A (our `InfiniCompiler` returns FX graph unchanged via `_compile_passthrough`) | `"vllm_ascend.compilation.compiler_interface.AscendCompiler"` |
| Fusion pass manager | none | `"vllm_ascend.compilation.graph_fusion_pass_manager.GraphFusionPassManager"` |
| `compilation_config.custom_ops` | `["all"]` | `["all"]` |

Ascend's platform hook injects:

```python
compilation_config.oot_compiler = cls.get_compile_backend()
# -> "vllm_ascend.compilation.compiler_interface.AscendCompiler"
```

and registers a pass-manager:

```
vllm-ascend/vllm_ascend/compilation/passes/
├── allgather_chunk_noop_pass.py        # communication elimination
├── allreduce_rmsnorm_fusion_pass.py    # fuse allreduce + rmsnorm
├── muls_add_pass.py
├── noop_elimination.py
├── norm_quant_fusion_pass.py           # fuse norm + quant
├── qknorm_rope_fusion_pass.py          # fuse rms_norm(Q) + rms_norm(K) + rope
├── sequence_parallelism.py
├── sequence_parallelism_moe.py
└── (plus) acl_graph.py                 # ACL-graph replay backend
```

Each pass uses `torch._inductor.pattern_matcher.PatternMatcherPass` to match multi-op subgraphs in the FX graph and replace them with a single fused C++ op (e.g. `torch.ops._C_ascend.npu_add_rms_norm_bias`, `torch.ops.npu.npu_fused_infer_attention_score`).

The fused replacements show up in the ascend cProfile as single dispatches, explaining:

- `torch._ops.npu.npu_fused_infer_attention_score` 2304 calls (one per attn layer per forward)
- `torch._ops.atb._npu_reshape_and_cache` 2304 calls
- `torch._ops._C_ascend.<various fused norms>` 72 calls

Whereas in infini the same operations show as three+ separate dispatches:

- `torch._ops.vllm.infini_unquantized_gemm` 18,496 calls
- `torch._ops.vllm.infini_add_rms_norm` 4,608 calls
- `torch._ops.vllm.infini_rotary_embedding_v2` 2,304 calls
- `torch._ops.vllm.infini_swiglu` 2,304 calls

## Revised options

The original F1/F2 plan (short-circuit `torch.ops.vllm.infini_*` dispatch to the underlying kernel) would save eager-mode Python overhead but **would not reduce dispatch count in the compiled FX graph** — Dynamo traces the custom-op call node regardless of the Python-side shortcut. F1/F2 alone won't close the 4.3x gap on graph mode; it would only help eager.

### Option G1 — Mirror ascend's fusion pass approach (big, the right lever)

Write inductor-style `PatternMatcherPass` passes for vllm-infini:

- `(rms_norm(x) + residual)` → already fused as `infini.ops.add_rms_norm`, but the FX graph has it split. Teach the pass to recognise the split pattern and replace with a single `infini.ops.add_rms_norm` call.
- `linear + rope` → find kernel. `infini.ops` doesn't have a fused version today.
- `linear + reshape_and_cache` → find kernel.

Plug these into our `InfiniCompiler._compile_passthrough` via `PatternMatcherPass.apply(graph)` before returning.

**Pros**: matches vllm-ascend's proven architecture. Addresses the root cause.

**Cons**: big lift. Each fusion pass is 100-500 lines of pattern-matching + kernel-wiring + tests. Needs new fused kernels in `infini.ops.*` (operator-side work). Mission is a 16 pp gap on graph mode; each pass is 1-3 pp.

### Option G2 — Bypass our `torch.ops.vllm.infini_*` registrations entirely at the FX graph level

Teach `InfiniCompiler` to rewrite the Dynamo FX graph: replace every `call_function` node targeting `torch.ops.vllm.infini_<op>` with a `call_function` node targeting `infini.ops.<op>` (pybind11 C++ entry) plus the wrapper prep (stream ptr, output alloc). Dynamo sees the final graph and the custom-op dispatch layer is removed.

**Pros**: single surgical FX-rewrite pass; eliminates the per-layer dispatch hop. Does NOT need new fused kernels — each op still runs standalone, we just drop one dispatcher.

**Cons**: couples our compiler to the exact shape of our FX call_function nodes. If Dynamo ever inlines the custom op wrapper differently, the pass mis-fires. Need fakes to stay so Dynamo can still trace.

**Expected savings**: dispatcher-hop is ~50% of per-op Python time at the FX-graph level (rough estimate). If F1/F2 couldn't do this from the eager side, doing it at the graph level could actually work. **But I need to measure before claiming this.**

### Option G3 — Accept the ceiling, ship the 92% eager result

Eager target is met. Document graph-mode as "architecturally blocked on a fusion-pass infrastructure we don't have" and stop. 63-66% of ascend in graph mode is still a respectable number given the structural gap.

## Recommendation

**G2 first, as a 1-2 day probe**. If it lands with correctness and measurable delta on graph mode, we close >10 pp cheaply. If it doesn't, we have clear evidence that G1 is the only path and can discuss whether to invest the larger effort.

**Skip F1/F2**: they would only benefit eager mode (which is already at 92%). Effort / payoff is bad.

## Time-box usage

Started at `T`, ~25 min elapsed. Answered in under time-box; no need to fall back to the bounded-probe plan.

## Unknowns not answered

1. Whether G2 actually reduces the FX-graph dispatch count in practice. Need a one-commit prototype that rewrites one op (e.g. `infini_unquantized_gemm`) and re-measures. ~2-4 hours work.
2. Whether the FX-rewrite approach plays nicely with NPUGraph capture / replay. Risk: a rewritten node might not be replay-safe if we lose the custom-op boundary markers.

## Commits

- This document: will be committed as `docs(perf): ...` alongside the message to team-lead.
