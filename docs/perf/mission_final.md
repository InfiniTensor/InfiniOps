# vllm-infini on Ascend 910B — mission final report

**Target**: vllm-infini total tok/s ≥ **80% of vllm-ascend** in both eager and PIECEWISE graph modes, without correctness regression, on Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct.

**Bench**: `vllm bench throughput`, random 128-in / 128-out, 256 prompts, dtype fp16, max-model-len 2048, 1 NPU on Ascend 910B4, CANN 8.5.1, container `infiniops-bench-ascend-v2`.

## Outcome

| Axis | 0.5B | 3B | Target | Result |
| --- | ---: | ---: | ---: | --- |
| **Eager** | **94.49%** | **95.22%** | ≥80% | **met with ~12 pp margin** |
| **Graph** | 67.28% | 71.51% | ≥80% | **short by 9-13 pp** |

**Partial success.** Eager is a real win with margin. Graph fell short — mid-mission the pivot from kernel-level wins to host-side wins ran into an architectural gap (vllm-ascend's FX fusion backend) that cannot be closed in this time-box. Mission ships with the eager outcome banked and the graph ceiling documented for a follow-up scoped project.

**Correctness** (greedy, fp16, 6-prompt token diff vs vllm-ascend): 3B 6/6 exact on eager and graph. 0.5B 5/6 on eager (pre-existing fp16 drift at token 57, not regressed). 0.5B 6/6 on graph.

## The four levers that moved the numbers

Net gain from baseline to final: 0.5B eager +23.7 pp, 0.5B graph +16.1 pp, 3B eager +16.1 pp, 3B graph +19.3 pp.

**1. Stream-pointer cache** (commit `c5593db`). cProfile traced ~27% of host wall time to `_stream.py:current_stream_ptr` being called ~326×/forward at ~21 us each. Module-level cache with per-forward invalidation on `GPUModelRunner.execute_model`. Kill-switch `INFINI_CACHE_STREAM=0`. Impact: +21 pp 0.5B eager, +22 pp 0.5B graph, +13 pp 3B eager, +12 pp 3B graph — the single biggest lever, got 3B eager past 80% on its own.

**2. G2: FX-graph dispatch rewrite** (commit `e05f613`). cProfile diff found infini doing 30,080 `_ops.__call__`/forward vs vllm-ascend's 6,976 (4.31×). Every `infini_<op>` went through `torch.ops.vllm.<op>` + an inner wrapper — two dispatcher hops. `vllm_infini/_direct_dispatch.py` rewrites FX `call_function` targets directly to pybind shims. Kill-switch `INFINI_DIRECT_DISPATCH=0`. Impact: ncalls 30,080 → 2,368 (12.7× fewer), +7.7 pp on 3B graph, strict improvement on all 4 axes, correctness unchanged.

**3. GatherV3 already-hoisted (audit, not new code)**. Team lead flagged 30.9 ms GatherV3 as the apparent rope-cos/sin hotspot. Audit found `ops/rotary_embedding.py` already runs `index_select` once per step via a weakref-based cache shared across all 36 layers (pre-existing before this mission). Operator's Task #29 confirmed infini and vllm-ascend at parity (1.12 ms / 100 calls vs 1.06 ms / 92 calls). No work needed; the 30.9 ms was stale pre-hoist data. Surfaced in audit, not shipped as new code.

**4. FX collapse pass `split_rope_collapse`** (commit `3d332cd`, opt-in). Designed to collapse the 36× `view → split → 3*getitem → rope → 2*getitem` chain. Shipped via `INFINI_FUSION_PASSES=split_rope_collapse`. Measured within noise (6,244 on / 6,222 off on 3B graph) because replacement has identical dispatch count (1 split + 3 getitem → 1 call_function + 3 getitem). Kept opt-in as reference code and pass-manager exercise; not default.

Also shipped: pass-manager scaffolding (commit `9b91b3f`) — `vllm_infini/compilation/pass_manager.py` + `INFINI_FUSION_PASSES` env var, empty default registry, runs in `InfiniCompiler._compile_passthrough` before the G2 rewrite. Lands the plumbing so the next fusion pass can be added in minutes.

## What we learned that the next team needs

Four durable nuggets, each saves days of reinvention. All saved as memory entries for future Claude sessions.

- **Dispatch-count asymmetry writeup** (`docs/perf/dispatch_count_mystery_2026-04-17.md`). The 30,080 vs 6,976 gap is not a vLLM-core issue — it's that vllm-ascend's compile backend runs 8 FX pattern-matcher fusion passes that collapse per-layer dispatch chains, and we run zero. That finding drove the G2 decision and scoped the structural ceiling below.
- **ATB `NormRopeReshape` is DeepSeek-MLA-only** (operator's #21 survey). Has QK-norm + RMS-norm built into the op definition; matches Qwen3/DeepSeek (`qk_norm=True`) but not Qwen2.5 (`qk_norm=False`). Ruled out one of the easier-looking fused-kernel paths. Don't re-open without a Qwen3 workload.
- **`aclnnRopeWithSinCosCache` hidden attrs** (memory `aclnn_rope_with_sin_cos_cache_hidden_attrs.md`). Task #22 wrapper failed with magnitude-8 output diffs because the public header silently hides 4 REG_OP-required attrs. Operator closed #22; don't retry without CANN vendor engagement for the full signature.
- **GatherV3 was a ghost hotspot**. The 30.9 ms number floated as a graph-gap target through several design docs. Real per-call time was 1.12 ms, total was pre-hoist stale. Lesson: re-slice msprof with `tests/decode_steady_state.py` (first-input-dim == batch_size filter) before trusting per-op deltas — prefill/warmup contamination is the default. Same lesson invalidated the earlier MatMulV2 +12% and greedy-sampler 27 ms claims. Saved as `feedback_measure_before_shipping.md`.

## Rejected levers (measured-before-shipped saved these)

- **F1 / F2 (drop `torch.ops.vllm.*` eager hop)** — eager was already 92-95% post-stream-cache; churn not worth it.
- **P-3 (FX `aten.to` / `aten.view` noop elimination)** — 0 matches on Qwen2.5 FX graph. All 36 `aten.to` are real bf16→fp16 casts; all 324 `aten.view` are real reshapes.
- **#27 hoist cos/sin gather** — already shipped pre-mission (weakref cache).
- **Env-flag sweep** (`INFINI_DECODE_ATTENTION=fa|pa_d2h_free`, `INFINI_USE_TORCHAIR=1`) — best case +2% (`fa` on 3B only); others regressed. Not a combinatorial lever.

## The graph ceiling — why stopping at 67-71% is structural

**Per-decode-step device time is at parity** (msprof decode-only slice: infini 11.47 ms vs ascend 11.63 ms on 3B graph). The entire 9-13 pp gap is host-side Python dispatch overhead.

vllm-ascend closes that gap with a **custom inductor-like compile backend plus 8 FX pattern-matcher fusion passes** (`vllm_ascend.compilation.compiler_interface.AscendCompiler`; passes: `qknorm_rope_fusion`, `norm_quant_fusion`, `allreduce_rmsnorm_fusion`, `muls_add_pass`, `noop_elimination`, `sequence_parallelism*`, `allgather_chunk_noop`, `split_qkv_fusion`). Each pass rewrites per-layer op chains into single fused `torch.ops._C_ascend.*` calls.

On Qwen2.5 specifically: `qknorm_rope_fusion` misses (no QK-norm), `allreduce_*` / `sequence_parallelism*` / `allgather_*` miss (TP=1), `norm_quant_fusion` misses (no quant), `noop_elimination` / `muls_add_pass` are genuine noops. The passes that *do* fire on Qwen2.5 in vllm-ascend are the ones targeting `rms_norm + qkv_proj` / `rms_norm + gate_up_proj` / `rope + reshape_and_cache` — and every one of them requires a fused kernel on the far side of the pass. No public aclnn/ATB API covers these fusions; we don't have the kernels, and the passes without kernels to call are not useful.

## Decision matrix for graph ≥80%

If the target remains binding, the work is:

| Lever | Effort | Who | Payoff |
| --- | --- | --- | --- |
| Port vllm-ascend's 8-pass FX fusion manager + compile backend | 2-3 weeks | vllm-infini | Infrastructure only; no throughput by itself |
| Fused `rms_norm + qkv_proj` AscendC kernel | 1-2 weeks | operator | ~3-5 pp graph, only with above |
| Fused `rms_norm + gate_up_proj` AscendC kernel | 1-2 weeks | operator | ~2-3 pp graph, only with above |
| Fused `rope + reshape_and_cache` (aclnn or AscendC) | 1 week | operator | ~2-4 pp graph; `aclnnRopeWithSinCosCache` is blocked on hidden attrs (see #22 memory) |
| Triton port of vllm-ascend's `qkv_rmsnorm_rope` kernel | 3-5 days | operator | Does not help Qwen2.5 — only Qwen3/DeepSeek |

**Minimum viable path to graph 80%**: compile-backend port + ≥2 of the fused kernels, ~4-6 weeks with operator engagement. Less than that won't close the gap. More than the target needs won't either — this is a compound investment, not a one-shot.

## Mission status

**Banked**: eager ≥80% on both Qwen2.5-0.5B and Qwen2.5-3B, correctness preserved, all optimizations kill-switchable via env vars.

**Not met**: graph ≥80% on either model. Ceiling at 67-72% is structural; closing it is a scoped multi-week project with operator engagement, not a continuation of this mission.

**Recommendation**: accept partial success, ship the work, re-scope graph-mode target as a separate project if still binding.

## Reproducibility

```bash
# Install (inside container infiniops-bench-ascend-v2).
cd /workspace/vllm-infini && pip install -e . --no-build-isolation

# Throughput.
VLLM_PLUGINS=infini python3 -m vllm.entrypoints.cli.main bench throughput \
  --model /workspace/models/Qwen/Qwen2.5-3B-Instruct \
  --dtype float16 --max-model-len 2048 \
  --dataset-name random --random-input-len 128 --random-output-len 128 \
  --num-prompts 256

# Correctness (greedy token diff).
python3 /tmp/correctness_check.py --model <path> --output-json /tmp/out_infini.json
python3 /tmp/diff_outputs.py /tmp/out_infini.json /tmp/out_ascend.json

# Env toggles.
INFINI_CACHE_STREAM=0                       # disable stream-ptr cache
INFINI_DIRECT_DISPATCH=0                    # disable G2 FX rewrite
INFINI_FUSION_PASSES=split_rope_collapse    # opt into the #28 pass
```
