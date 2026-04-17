# Scoped G1 fusion design — close remaining graph-mode gap

**Status**: design doc, not code. Submitted for team-lead review before implementation starts. Operator survey filed as Task #19 in parallel.

## Context

After G2 (`e05f613`), graph mode sits at:

- 3B graph: **71.51%** of vllm-ascend (target 80%, gap ~8.5 pp)
- 0.5B graph: 67.28% (target 80%, gap ~12.7 pp)

G2 removed the `torch.ops.vllm.infini_*` dispatcher hop (12.7× fewer `_ops.__call__`). Remaining gap must come from actual op-count in the graph: per-layer gemm / norm / rope / swiglu are still separate FX nodes, each with its own pybind-entry cost, output-tensor allocation, and per-op scheduling overhead.

G1's premise: **fewer, bigger ops per layer**. Ascend does this with 8 FX `PatternMatcherPass` classes. Full port is a multi-week lift; this doc scopes which subset to actually port for Qwen2.5 decode on single NPU.

## Model-specific analysis: Qwen2.5

Per-layer FX op sequence in `Qwen2DecoderLayer.forward`:

```
1.  x, residual = input_layernorm(x, residual)        # add_rms_norm
2.  qkv = qkv_proj(x)                                 # gemm
3.  q, k, v = qkv.split(...)
4.  q, k = rotary_emb(positions, q, k)                # apply_rotary_pos_emb
5.  y = unified_attention_with_output(q, k, v, ...)
6.  y = o_proj(y)                                     # gemm
7.  y, residual = post_attention_layernorm(y, residual)  # add_rms_norm
8.  gate, up = (gate_proj(y), up_proj(y))             # 2 gemms  (OR 1 merged gemm)
9.  mlp_in = silu_and_mul(concat(gate, up))           # silu_and_mul
10. y = down_proj(mlp_in)                             # gemm
```

Key differences from vllm-ascend's `qknorm_rope_fusion_pass` target (Qwen3/DeepSeek models):

- **Qwen2.5 has no Q/K norm**: `qk_norm` attribute exists but defaults `False`. So ascend's `qknorm_rope_fusion_pass` does not match our FX graph at all.
- The `gate_proj`/`up_proj` pair may already be merged into a single gemm via `MergedColumnParallelLinear` in vLLM. Confirmed by grep on the model (line 531: `"qkv_proj": ["q_proj", "k_proj", "v_proj"]`). Need to verify in our FX dump whether infini's graph shows 1 or 2 MLP-input gemms.

Per-forward dispatch inventory (from cProfile of G2-enabled run, 3B graph, 64 forwards):

| Op family | ncalls | per-forward | Fusion candidate? |
| --- | ---: | ---: | --- |
| infini_unquantized_gemm (direct) | 18,496 | 289 | gate+up merged? qkv split fused into rope? |
| infini_add_rms_norm | 4,608 | 72 | pair = input + post-attn per layer (2 × 36) ✓ |
| apply_rotary_pos_emb | 2,304 | 36 | fuse with q/k slicing ✓ |
| silu_and_mul | 2,304 | 36 | fuse into gate/up gemm? ✗ (no aclnn API) |
| unified_attention_with_output | 2,304 | 36 | already fused (vLLM wraps attention impl) |
| reshape_and_cache | 2,304 | 36 | already a single ATB op |

## Candidate passes (scoped for Qwen2.5, TP=1 decode)

### P-1: `split_rope_fusion_pass` (highest ROI)

Pattern:

```python
def pattern(qkv, positions, cos_sin_cache, head_dim):
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    q_rope, k_rope = torch.ops.vllm.infini_rotary_embedding_v2(
        positions, q, k, cos_sin_cache, head_dim, is_neox_style
    )
    return q_rope, k_rope, v
```

Replacement:

```python
def replacement(qkv, positions, cos_sin_cache, head_dim):
    q_rope, k_rope, v = torch.ops.vllm.infini_qkv_split_rope(
        qkv, positions, cos_sin_cache,
        q_hidden_size=q_size, kv_hidden_size=kv_size,
        head_dim=head_dim, is_neox_style=is_neox_style,
    )
    return q_rope, k_rope, v
```

Occurrences per forward: **36** (one per layer).

Reduction: eliminates the `aten.split` / 3-way `aten.slice` triplet + `infini_rotary_embedding_v2` as distinct nodes, collapsing to one `infini_qkv_split_rope`. Expected dispatch count drop: 36 * (3 slices + 1 rope - 1 fused) = 108/forward, about 4% of remaining graph dispatches.

Operator-side requirement: need a fused `infini.ops.qkv_split_rope` kernel. **Operator survey (#19) reports availability.**

Expected payoff if kernel exists: 2-4 pp on 3B graph.

### P-2: `add_rms_norm_concat_pass` (medium ROI, no new kernel)

Pattern: two consecutive `add_rms_norm` calls (input + post-attn) per layer can share buffer allocation and stream setup.

This is NOT a kernel fusion — `infini.ops.add_rms_norm` already exists. The optimization is **eliminating `torch.empty()` calls** between the two norms by reusing the output buffer. Small effect per-op but compounds over 72 calls/forward.

Alternative framing: keep the ops separate but pre-allocate per-layer output buffers once (weakref cache, already done for rope). Expected payoff: 1-2 pp.

**Should this be G1 or an eager-side micro-opt?** Because it doesn't change FX structure, maybe it belongs in `ops/layernorm.py` as a same-commit change when operator confirms kernel stability. Open question for review.

### P-3: `noop_elimination` (free, small)

Ascend's pass drops obvious no-op FX nodes (e.g., `aten.view` with identical shape, `aten.to` with matching dtype). Python-side overhead per no-op is real (10+ us per dispatch). Occurrence count in our graph TBD — need FX dump.

Expected payoff: <1 pp (unless count turns out high).

## Out-of-scope (explicitly skipped for this round)

- `allreduce_rmsnorm_fusion_pass`: TP-only, we're TP=1.
- `allgather_chunk_noop_pass`: TP/SP-only.
- `sequence_parallelism*`: SP not in our bench.
- `norm_quant_fusion_pass`: quantization not in our bench.
- `muls_add_pass`: investigate only if P-1 + P-3 land and we're still short.

## Pass manager skeleton

Mirror ascend's layout but trimmed:

```
vllm-infini/vllm_infini/compilation/
├── __init__.py
├── pass_manager.py             # collects passes, applies in order
├── base_pattern.py             # helper for PatternMatcherPass-derived classes
└── passes/
    ├── __init__.py
    ├── split_rope_fusion.py   # P-1
    └── noop_elimination.py     # P-3
```

Wire-in (in `_compiler.py._compile_passthrough`, after `maybe_rewrite_infini_dispatches`):

```python
from vllm_infini.compilation.pass_manager import apply_fusion_passes
graph = apply_fusion_passes(graph)
```

`apply_fusion_passes` reads `INFINI_FUSION_PASSES` (default=`"all"`; `=0` or `""` disables all; `=split_rope,noop` enables specific ones).

## Kill-switch and rollback

- `INFINI_FUSION_PASSES=0` — disable all passes.
- `INFINI_FUSION_PASSES=split_rope` — enable only P-1 (useful for bisection).
- Each pass logs `logger.info("fused N <pattern>")` at INFO level so we can verify it ran.

## Measurement plan

Per-pass commit cycle:

1. Code the pass.
2. Correctness gate: `/tmp/correctness_check_graph.py` on both 3B and 0.5B, diff vs vllm-ascend outputs. Require 6/6 exact token match on 3B (baseline). Allow one divergence on 0.5B within the fp16-noise pattern seen before (tolerance: same divergence count as `e05f613` baseline).
3. cProfile: compare `_ops.__call__` ncalls pre/post. Expected delta recorded in pass's docstring.
4. Throughput: full bench matrix (0.5B + 3B, eager + graph). Record in `docs/perf/e2e_progress.md`.
5. If ratio moves <1 pp on 3B graph despite call count dropping as expected → pivot (don't finish the rest of the passes).

Measurement baseline (post-G2):

| Model | Mode | tok/s | vs ascend |
| --- | --- | ---: | ---: |
| 0.5B | graph | 10,445 | 67.28% |
| 3B   | graph |  7,258 | 71.51% |

Target gates:

- ≥80% on 3B graph after P-1 lands → mission complete, backport eager results to report.
- 73–79% → continue with P-3 and (if operator scopes it) any cheap P-2.
- ≤72% → fusion passes aren't the lever; halt G1 and ship G3 with full documentation.

## Operator-side dependency (Task #19)

Fused kernel needs that operator team must confirm:

1. **`infini.ops.qkv_split_rope`** — takes `(qkv_tensor, positions, cos_sin_cache, q_hidden, kv_hidden, head_dim, is_neox_style)`, returns `(q_rope, k_rope, v)`. Ideally backed by an ATB/aclnn fused API if one exists; otherwise an AscendC custom kernel.

If the operator survey reports "no fused API, custom kernel required", the design decision becomes: (a) G1 is no longer a 3-day probe — scope slips multi-day into operator's critical path; revisit G3 ship. (b) Ship P-3 alone (not blocked on new kernels), measure, and only commission the custom kernel if it would move the remaining needle.

## Risk summary

- **Biggest unknown**: whether `torch._inductor.pattern_matcher.PatternMatcherPass` plays cleanly with our `_compile_passthrough` (non-aot_autograd) path. Ascend uses it inside an inductor-like backend; we might need to wrap our graph in a compatible interface. If this turns into a rabbit hole, time-box the wire-up separately.
- **NPUGraph capture correctness**: if P-1 replaces multiple FX nodes with a single `torch.ops.vllm.infini_qkv_split_rope` custom-op node, G2's direct-dispatch rewrite in `_direct_dispatch.py` needs to also know about this new op name (otherwise we'd get back the dispatcher hop for the fused op).
- **Fused-kernel failure modes**: new AscendC kernels have a history of bugs (per memory `matmul_kernel_ceiling`). Plan extra buffer on correctness diff cycle.

## Ask for team-lead review

Specifically:

1. Approve scoping to P-1 + P-3 for the first round; defer P-2 to operator availability.
2. Confirm it's acceptable to block on Task #19 before writing pass code (can't write P-1 without knowing what fused op to call).
3. Approve the `INFINI_FUSION_PASSES` env-var design (matches existing `INFINI_DIRECT_DISPATCH` / `INFINI_CACHE_STREAM` / `INFINI_DECODE_ATTENTION` kill-switch style).
4. Any objection to measurement going through the existing `docs/perf/e2e_progress.md` row cadence vs a separate G1 sub-report?
