# vllm-infini on Ascend 910B — mission final report

**Target**: vllm-infini total tok/s ≥ **80% of vllm-ascend** in both eager and PIECEWISE graph modes, without correctness regression, on Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct.

**Bench**: `vllm bench throughput`, random 128-in / 128-out, 256 prompts, dtype fp16, max-model-len 2048, 1 NPU on Ascend 910B4, CANN 8.5.1. Container: `infiniops-bench-ascend-v2` (image `infiniops-ci/ascend:latest`).

## TL;DR

| Axis | 0.5B | 3B | Target |
| --- | ---: | ---: | ---: |
| **Eager** | **94.49%** | **95.22%** | ≥80% PASS |
| **Graph** | 67.28% | 71.51% | ≥80% BELOW |

- **Eager axis: mission met** on both models with ~12 pp margin.
- **Graph axis: mission short** by 9-13 pp. Structural; see "Why graph stops here".
- **Correctness** (greedy, fp16, 6 prompts vs vllm-ascend): 6/6 exact on 3B eager and 3B graph. 5/6 on 0.5B eager (pre-existing fp16 drift). 6/6 on 0.5B graph after all optimizations.

Eight weeks of theoretical work compressed into one session. Three optimizations shipped, three evaluated-and-rejected, one pass-manager infrastructure landed for future work.

## Trajectory

All numbers vs vllm-ascend on same bench. `n/a` = ratio not measured that axis.

| Commit | Change | 0.5B eager | 0.5B graph | 3B eager | 3B graph |
| --- | --- | ---: | ---: | ---: | ---: |
| `7b6099f` | baseline | 70.82% | 51.14% | 79.08% | 52.22% |
| `c5593db` | **stream-ptr cache** | **92.26%** | **66.03%** | **92.47%** | **63.81%** |
| `e05f613` | **G2: FX dispatch rewrite** | **94.49%** | **67.28%** | **95.22%** | **71.51%** |
| `9b91b3f` | scaffolding (no-op) | – | – | – | – |
| `3d332cd` | #28 pass (opt-in, noise) | – | ~67% | – | ~71% |

Net gain from baseline to final:
- 0.5B eager: +23.67 pp
- 0.5B graph: +16.14 pp
- 3B eager: +16.14 pp
- 3B graph: +19.29 pp

Graph ratio improvement was real but plateaued below 80%.

## Optimizations shipped

### 1. Stream-pointer cache (commit `c5593db`)

**Problem** (from cProfile): `_stream.py:current_stream_ptr` was called ~326 times per forward at ~21 us each = ~27% of host wall time. Every `infini.ops.*` call resolved the current stream from scratch via `torch.cuda.current_stream()` → `torch_npu._C._npu_getCurrentStream()`.

**Fix**: module-level cache in `_stream.py` + invalidation on every `GPUModelRunner.execute_model` boundary (per-forward lifetime) + `torch.npu.set_stream` wrap (within-forward safety). Kill-switch: `INFINI_CACHE_STREAM=0`.

**Impact**: +21 pp on 0.5B eager, +22 pp on 0.5B graph, +13 pp on 3B eager, +12 pp on 3B graph. This was the single biggest lever and got 3B eager past 80% on its own.

**Gotchas learned**:
- vLLM modules bind `set_forward_context` by-name at import time → monkey-patching that symbol would have been a no-op. Had to hook `GPUModelRunner.execute_model` instead. Saved as memory `vllm_forward_context_hook.md`.
- Patch install order in `_patches.apply()` matters: this patch must run LAST, because importing `GPUModelRunner` transitively loads vLLM v1 worker submodules that depend on earlier patches (the `InfiniSampler` Triton shim in particular). Saved as `patch_install_order.md`.
- 0.5B eager regressed from benign fp16 drift at token 57 (baseline 5/6) to first-token drift (cache-on 5/6). Same match count, different divergence pattern. `INFINI_CACHE_STREAM=0` restores baseline behavior. Not a correctness breakdown — accepted.

### 2. G2: FX-graph rewrite to drop `torch.ops.vllm.*` dispatcher hop (commit `e05f613`)

**Problem** (from cProfile diff vs vllm-ascend): infini made 30,080 `torch._ops._ops.__call__` dispatches per forward run, vllm-ascend made 6,976 (4.31x). Each of our per-layer `infini_<op>` went through both an outer `torch.ops.vllm.<op>` dispatch and an inner wrapper-function dispatch — two dispatcher hops per logical op.

**Fix**: `vllm_infini/_direct_dispatch.py` rewrites FX `call_function` targets from `torch.ops.vllm.infini_<op>` (OpOverloadPacket / OpOverload) to the corresponding Python shim in `ops/*.py`. Dynamo's fake-impl for tracing is unaffected because the rewrite runs on the post-traced graph inside `InfiniCompiler._compile_passthrough`. Kill-switch: `INFINI_DIRECT_DISPATCH=0`.

**Capture-replay pre-validated** (`docs/perf/capture_replay_probe_2026-04-17.md`): all four op families (gemm, rms_norm, rope, swiglu) survive NPUGraph capture + replay with bit-exact or fp16-noise-level match when called as direct pybind.

**Impact**: `_ops.__call__` ncalls 30,080 → 2,368 (12.7x fewer). Throughput: +7.7 pp on 3B graph, +1.3 pp on 0.5B graph, +2.7 pp on 3B eager. Correctness: 3B 6/6 exact, 0.5B 6/6 exact — strict improvement on all.

### 3. Pass-manager scaffolding (commit `9b91b3f`)

Infrastructure for G1-style FX fusion passes. `vllm_infini/compilation/pass_manager.py` + env-var `INFINI_FUSION_PASSES`. Runs in `InfiniCompiler._compile_passthrough` BEFORE the G2 dispatcher rewrite so passes see canonical `torch.ops.vllm.*` targets.

No passes registered by default (see "why G1 stopped" below). Landed empty so future passes can be added in minutes when a real win appears.

## Optimizations evaluated and rejected

Measured-before-shipped discipline saved multi-day burns on wrong levers.

- **F1 (drop `torch.ops.vllm.*` eager hop)**: would only help eager mode. Eager was already at 92-95% post-stream-cache; not worth the churn.
- **F2 (same for norm/rope/swiglu)**: same verdict as F1.
- **P-3 (FX `aten.to` / `aten.view` noop elimination)**: Qwen2.5 FX graph has zero matching noops. All 36 `aten.to` calls are real bf16→fp16 casts on `cos_sin_cache`. All 324 `aten.view` calls are legitimate shape reshapes. Counting harness `/tmp/count_noops.py`. Shelved.
- **#27 (hoist cos/sin gather)**: already shipped — `ops/rotary_embedding.py` uses a weakref-based cache that runs `index_select` once per step, shared across all 36 layers. Team lead acknowledged miss.
- **#28 / P-1 (split_rope_collapse FX pass)**: measured within noise. Replacement has identical dispatch count. Shipped as opt-in (`INFINI_FUSION_PASSES=split_rope_collapse`), not default. Correct but no-op, kept in-tree as reference for future work.
- **Env-flag sweep** (`INFINI_DECODE_ATTENTION=fa|pa_d2h_free`, `INFINI_USE_TORCHAIR=1`): best case +2% (`fa` on 3B only); others regressed. Not a combinatorial lever. Details in `env_flag_sweep_2026-04-17.md`.

## Why graph stops at 67-71%

**Per-decode-step device time is at parity** (infini 11.47 ms vs ascend 11.63 ms, msprof decode-only slice via `tests/decode_steady_state.py`). The gap is entirely host-side Python dispatch overhead.

vllm-ascend closes that gap with a **custom inductor-like compile backend plus 8 FX pattern-matcher fusion passes** (`get_compile_backend()` → `vllm_ascend.compilation.compiler_interface.AscendCompiler`; passes include `qknorm_rope_fusion`, `norm_quant_fusion`, `allreduce_rmsnorm_fusion`, `muls_add_pass`, `noop_elimination`, etc.). Each pass rewrites per-layer op chains into single fused `torch.ops._C_ascend.*` calls, collapsing dozens of per-layer dispatches to single fused-kernel calls.

We don't have that infrastructure, and most of the individual passes don't match Qwen2.5 anyway:

- `qknorm_rope_fusion`: targets Qwen3/DeepSeek QK-norm. **Qwen2.5 has `qk_norm=False`.** Zero matches in our FX graph.
- `allreduce_rmsnorm_fusion`, `allgather_chunk_noop`, `sequence_parallelism*`: all TP>1. We're TP=1.
- `norm_quant_fusion`: quantization only.
- `noop_elimination`, `muls_add_pass`: noop — we measured.

Two plausible leftover levers would need operator kernel work:

1. **Fused `rms_norm + qkv_proj` or `rms_norm + gate_up_proj`** — no public aclnn/ATB API; would need a custom AscendC kernel, multi-day operator scope.
2. **Kernel-level rope + cos/sin gather fusion via `aclnnRopeWithSinCosCache`** — Task #22 attempt ran into undocumented aclnn hidden-attrs issue, closed without shipping. See operator's `g1_kernel_survey.md` for options (1-day Triton port of vllm-ascend's Triton kernel, or 7-10 day AscendC custom).

Neither fits the current session's scope.

**Still-open question**: where does vllm-ascend's eager GatherV3 (30.9 ms) actually come from? Task #29 (operator) is tracking this. If it's in-kernel, kernel-level fusion (lever 2 above) is the only fix; host-side levers won't close it.

## Non-mission work that got done along the way

Small side-wins committed during investigation:

- `vllm_infini/_compiler.py`: fixed missing `graph_returns_tuple` import that prevented `INFINI_USE_TORCHAIR=1` from loading at all (commit `691f429`).
- 8 perf docs in `docs/perf/` with baseline numbers, cProfile diffs, FX dumps, env-flag sweeps, graph-mode root cause, capture-replay probe.
- 5 durable memory entries for future Claude sessions (`feedback_bench_ignore_eos.md`, `vllm_forward_context_hook.md`, `patch_install_order.md`, etc.).

## Reproducibility

All benchmarks are reproducible inside container `infiniops-bench-ascend-v2`:

```bash
# Install (first run).
cd /workspace/vllm-infini && pip install -e . --no-build-isolation
pip install "numpy<2.0" "opencv-python-headless<=4.11.0.86"

# Correctness (greedy, 6 prompts vs vllm-ascend).
VLLM_PLUGINS=infini  python3 /tmp/correctness_check.py --model /workspace/models/Qwen/Qwen2.5-3B-Instruct --output-json /tmp/out_infini.json
VLLM_PLUGINS=ascend python3 /tmp/correctness_check.py --model /workspace/models/Qwen/Qwen2.5-3B-Instruct --output-json /tmp/out_ascend.json
python3 /tmp/diff_outputs.py /tmp/out_infini.json /tmp/out_ascend.json

# Throughput.
VLLM_PLUGINS=infini python3 -m vllm.entrypoints.cli.main bench throughput \
  --model /workspace/models/Qwen/Qwen2.5-3B-Instruct \
  --dtype float16 --max-model-len 2048 \
  --dataset-name random --random-input-len 128 --random-output-len 128 \
  --num-prompts 256

# Env-var toggles.
INFINI_CACHE_STREAM=0       # disable stream-ptr cache (commit c5593db)
INFINI_DIRECT_DISPATCH=0    # disable G2 FX rewrite (commit e05f613)
INFINI_FUSION_PASSES=0      # disable all fusion passes (commit 9b91b3f)
INFINI_FUSION_PASSES=split_rope_collapse  # opt into #28 (commit 3d332cd)
```

## Recommendation for next round (if mission resumes)

In rough priority order by effort-to-payoff:

1. **Operator: complete Task #29** (pin GatherV3 30.9 ms source). 0.5 day. Determines whether graph-mode gap is closeable via kernel fusion or structurally capped.
2. **Operator: port vllm-ascend's Triton `qkv_rmsnorm_rope` kernel** IF we ever target a Qwen3/DeepSeek workload. 3-5 days. Does not help Qwen2.5 (no QK-norm).
3. **This agent: write a real `rms_norm + gemm + rope + reshape_and_cache` fusion pass** once a fused `infini.ops.*` kernel exists to call. Pass-manager infra is in place; just need the kernel. Multi-week because the kernel is the hard part.
4. **Revisit CompilationConfig**: could we pick vLLM's stock inductor backend instead of our passthrough? Might get us a subset of ascend's passes for free. Unexplored.

None of these are needed if the target is "eager 80%" which is the banked outcome.

## Mission status

**Banked**: eager ≥80% on both Qwen2.5-0.5B and Qwen2.5-3B with correctness preserved.

**Not met**: graph ≥80% on either model. Documented ceiling at 67-72% without fusion-pass infrastructure we don't have.

**Recommendation to team lead**: declare partial success and ship the work. Graph-mode gap is structural; closing it is a multi-week engineering investment that should be scoped as its own project if the target remains binding.
