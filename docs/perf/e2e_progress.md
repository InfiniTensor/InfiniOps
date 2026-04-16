# E2E Throughput Progress — vllm-infini vs vllm-ascend on Ascend 910B

Target: vllm-infini total tok/s >= 80% of vllm-ascend total tok/s, for **both**
eager and PieceWise (graph) modes, **without** correctness regression.

Benchmark: `vllm bench throughput`, random dataset, 128 in / 128 out, 256
prompts, dtype float16, max-model-len 2048. One NPU (device 1) on Ascend
910B4, CANN 8.5.1, container `infiniops-bench-ascend-v2`.

## Trajectory

Columns: total tokens per second (infini / ascend), ratio, and notes.

| Date | Commit (vllm-infini) | Model | Mode | infini tok/s | ascend tok/s | Ratio | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-04-17 | `7b6099f` | 0.5B | eager     |  7,188.0 | 10,150.9 | 70.82% | Baseline. Correctness PASS. |
| 2026-04-17 | `7b6099f` | 0.5B | piecewise |  7,940.2 | 15,525.2 | 51.14% | Baseline. infini graph speedup only 1.10x vs ascend 1.53x. |
| 2026-04-17 | `7b6099f` | 3B   | eager     |  5,290.7 |  6,690.4 | 79.08% | Baseline. One pp below target — easiest to clear first. |
| 2026-04-17 | `7b6099f` | 3B   | piecewise |  5,299.1 | 10,147.6 | 52.22% | Baseline. infini graph speedup ~1.00x vs ascend 1.52x. |

## Status vs target

- **Eager**: 3B at 79% (essentially at target), 0.5B at 71% (below).
- **Graph**: both models ~51-52% — far below 80%.

## Critical finding (2026-04-17): the gap is host-side, not kernel

Re-sliced the msprof data to decode-only steady-state (`tests/decode_steady_state.py`
with first-input-dim == batch_size filter):

| Mode | infini per-decode-step (ms) | ascend per-decode-step (ms) | Ratio |
| --- | ---: | ---: | ---: |
| 3B eager | 11.62 | 11.44 | 1.02x |
| 3B graph | 11.47 | 11.63 | 0.99x |

**Per-step device time is effectively identical.** The 21-48% e2e gap is
**entirely host-side** (Python scheduling / metadata prep / launch pipeline /
async stream layout).

What this invalidates from the earlier backlog:

- ~~MatMulV2 +12%~~: actually +1% per decode call (65.4 vs 64.6 us). Delta was
  contaminated by prefill+warmup ops. (See Task #10 handoff to `operator`.)
- ~~Greedy-sampler waste (27 ms)~~: those ops fire during graph-capture warmup
  for a 256-row dummy batch, not per-step decode. (See
  `sampler_investigation_2026-04-17.md`.)

## Revised headline optimization backlog

- **P0**: CPU-side profile (`py-spy record` / `cProfile`) of
  `vllm bench throughput` on both plugins to find the exact Python hotspot.
  Device time is known to be a non-issue. See
  `graph_mode_root_cause_2026-04-17.md`.
- **P1**: move decode-path `cu_seqlens` cumsum to CPU in
  `vllm-infini/vllm_infini/attention/metadata.py` (already pinned CPU tensors
  exist for `pa_d2h_free` mode). Avoid per-step `torch.cumsum` on device.
- **P1**: try running exponential-random on a side stream (as
  `vllm_ascend/sample/sampler.py` does) so RNG overlaps compute.
- **P2 (operator)**: decode-time ATB/ACLNN variants that consume a device
  tensor for sequence lengths so we can graph-capture the full decode step.
  Our current PIECEWISE is forced because of per-call `aclIntArray*` baking.

## Env-flag sweep (2026-04-17)

See `env_flag_sweep_2026-04-17.md`.

| Config (3B graph) | tok/s | vs default |
| --- | ---: | ---: |
| default (`pa`) | 5,299.1 | 100.0% |
| `INFINI_DECODE_ATTENTION=fa` | 5,405.5 | **+2.0%** (take on 3B) |
| `INFINI_DECODE_ATTENTION=pa_d2h_free` | 4,994.0 | -5.8% |
| `INFINI_USE_TORCHAIR=1` | 4,372.5 | -17.5% |

Side fix: `vllm_infini/_compiler.py` was missing `graph_returns_tuple` import —
needed for `INFINI_USE_TORCHAIR=1` to load at all.

Detailed per-op data: see `e2e_baseline_eager_2026-04-17.md`,
`e2e_baseline_piecewise_2026-04-17.md`, and
`graph_mode_root_cause_2026-04-17.md`.
