# Env-flag sweep — vllm-infini graph mode

Reproduction of the `INFINI_DECODE_ATTENTION` / `INFINI_USE_TORCHAIR` variants
under the same e2e benchmark as `e2e_baseline_piecewise_2026-04-17.md`.

Same dataset: `vllm bench throughput`, random 128/128 in/out, 256 prompts,
dtype fp16, `--max-model-len 2048`, 1 NPU (device 1), PIECEWISE mode (no
`--enforce-eager`), 910B4, CANN 8.5.1, container `infiniops-bench-ascend-v2`.

## Results

| Model | Config | total tok/s | vs default | vllm-ascend | Ratio vs ascend |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5B | default (`pa`)            |  7,940.2 | 100.0% | 15,525.2 | 51.1% |
| 0.5B | `INFINI_DECODE_ATTENTION=fa`          |  7,736.7 |  97.4% | 15,525.2 | 49.8% |
| 3B   | default (`pa`)            |  5,299.1 | 100.0% | 10,147.6 | 52.2% |
| 3B   | `INFINI_DECODE_ATTENTION=fa`          |  5,405.5 | **+2.0%** | 10,147.6 | 53.3% |
| 3B   | `INFINI_DECODE_ATTENTION=pa_d2h_free` |  4,994.0 |  -5.8% | 10,147.6 | 49.2% |
| 3B   | `INFINI_USE_TORCHAIR=1`               |  4,372.5 | -17.5% | 10,147.6 | 43.1% |

## Key observations

1. **`INFINI_DECODE_ATTENTION=fa` is a small win on the 3B model (+2.0%) but a minor regression on 0.5B (-2.6%).** Not flip-the-chart worthy; still far from the 80% target.
2. **`pa_d2h_free` is a regression**, despite the design claim of eliminating per-layer D2H sync. This suggests the kernel variant itself is slower for the caller-provided host-tensor path, or the overhead of maintaining those CPU-side tensors outweighs the avoided D2H.
3. **`INFINI_USE_TORCHAIR=1` is badly regressed (-17.5%).** Torchair adds compilation cost that is not amortized over 256 short requests. Also: torchair was broken on entry (import bug) — see "side fix" below.
4. None of the env-flag combinations close the graph-mode gap.

## Side fix during this investigation

`vllm_infini/_compiler.py` was crashing with `NameError: name 'graph_returns_tuple' is not defined` whenever `INFINI_USE_TORCHAIR=1` was set. The symbol was used but never imported. Fixed by adding it to the existing `from torch._inductor.compile_fx import (...)` block. Reproducibility of torchair numbers above **requires** this fix.

## Recommendation

Drop env-flag tuning as a near-term lever. The ~1.5x graph-mode gap is a structural issue (per-step attention eager + per-op dispatch cost), not a flag-switching issue. Focus on Task #8 (understand vllm-ascend's graph-mode speedup source).

One small gain to take: on production 3B eager/decode workloads, set `INFINI_DECODE_ATTENTION=fa` by default (gives +2% on 3B with no downside observed). Verify on 7B-class models before pinning.

## Commands used

```bash
INFINI_DECODE_ATTENTION=fa            VLLM_PLUGINS=infini vllm bench throughput ...
INFINI_DECODE_ATTENTION=pa_d2h_free   VLLM_PLUGINS=infini vllm bench throughput ...
INFINI_USE_TORCHAIR=1                 VLLM_PLUGINS=infini vllm bench throughput ...
```

All JSONs in the container: `/tmp/bench_infini_graph_3b_{fa,pa_d2h_free,torchair}.json`, `/tmp/bench_infini_graph_0p5b_fa.json`.
