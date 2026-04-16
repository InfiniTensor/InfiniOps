# Graph-mode gap root cause — device time is not the problem

## TL;DR

**Per decode-step device time is essentially identical between vllm-infini and
vllm-ascend** — 11.5 ms vs 11.6 ms. The entire throughput gap is
**host-side overhead** (Python scheduling, metadata building, launch pipeline),
not kernel compute.

Evidence (Qwen2.5-3B, 8 prompts, 32 output tokens, msprof device-time,
decode-only ops filtered by `first input dim == batch_size == 8`):

| Run | Decode steps | Total decode device time (ms) | Per-step (ms) |
| --- | ---: | ---: | ---: |
| infini **eager** 3B   | 43 | 501.5 | 11.62 |
| ascend **eager** 3B   | 40 | 459.0 | 11.44 |
| infini **graph** 3B   | 63 | 722.5 | 11.47 |
| ascend **graph** 3B   | 49 | 574.3 | 11.63 |

Per-step ratio infini/ascend: eager 1.02x, graph 0.99x. **Within measurement noise.**

Yet at the throughput level, eager infini is ~79% of ascend and graph infini
is ~52% of ascend. The delta must be non-device-bound.

## Detailed decode-only kernel counts and timings (3B, graph mode)

### vllm-infini graph (decode-only, batch=8)

| OP | Count | Total (ms) | % | Avg (us) |
| --- | ---: | ---: | ---: | ---: |
| MatMulV2                  | 9072 | 589.7 | 81.6% | 65.0 |
| PagedAttentionMaskNdKernel| 2145 |  52.3 |  7.2% | 24.4 |
| SwiGlu                    | 2253 |  19.2 |  2.7% |  8.5 |
| Slice                     | 6687 |  17.1 |  2.4% |  2.6 |
| AddRmsNorm                | 4505 |  15.1 |  2.1% |  3.3 |
| AtbRopeKernel             | 2253 |  11.5 |  1.6% |  5.1 |
| ArgMaxV2                  |   61 |   9.1 |  1.3% | 149.6 |
| ReshapeAndCacheNdKernel   | 2145 |   6.3 |  0.9% |  2.9 |

### vllm-ascend graph (decode-only, batch=8)

| OP | Count | Total (ms) | % | Avg (us) |
| --- | ---: | ---: | ---: | ---: |
| MatMulV2                  | 7110 | 476.1 | 82.9% | 67.0 |
| FusedInferAttentionScore  | 1694 |  41.2 |  7.2% | 24.3 |
| SwiGlu                    | 1765 |  15.0 |  2.6% |  8.5 |
| AddRmsNormBias            | 3530 |  12.3 |  2.1% |  3.5 |
| Slice                     | 3634 |   9.3 |  1.6% |  2.6 |
| ArgMaxV2                  |   49 |   7.2 |  1.3% | 146.8 |
| _triton_rope              | 1766 |   6.2 |  1.1% |  3.5 |
| ReshapeAndCacheNdKernel   | 1694 |   5.1 |  0.9% |  3.0 |

Key per-call comparisons (decode only):

| Op | infini avg (us) | ascend avg (us) | Gap |
| --- | ---: | ---: | ---: |
| MatMulV2            | 65.0 | 67.0 | **-3.0% (infini wins)** |
| Attention decode    | 24.4 (PA) | 24.3 (FIA) | parity |
| SwiGlu              |  8.5 |  8.5 | parity |
| Add+RmsNorm         |  3.3 |  3.5 | infini wins |
| RoPE apply          |  5.1 (ATB) |  3.5 (triton) | **+46% (infini loses)** |
| ReshapeAndCache     |  2.9 |  3.0 | parity |

Only `AtbRopeKernel` is slower per-call on infini (+46%), but total RoPE time is
11.5 ms vs 6.2 ms — delta of only 5 ms out of 722 ms. Not load-bearing.

## Why was MatMulV2 reported as 12% slower earlier?

Earlier analysis compared total `MatMulV2` time across the whole profile,
which mixed prefill (very long sequences, `MatMulV2` avg > 100 us due to
large shapes) and warmup iterations. On decode-only slices the per-call time
is **within 3%** and can even favour infini.

**Takeaway**: total-op-time comparisons are dangerous when the workload has a
mixed phase (prefill + decode + warmup). Always slice by phase.

## What this implies for optimization strategy

Device time is essentially spent. Further kernel-level wins on infini decode
ops will not move the e2e needle materially. **The gap is host-side:**

1. **Kernel-launch count asymmetry**: at steady state infini and ascend issue
   roughly the same per-step launches, but the non-steady-state wrapper
   (metadata prep, sampler dispatch, next-step preparation) may take 2-3x more
   CPU time on infini. This needs a Python-level profile (cProfile / py-spy),
   not an NPU profile.
2. **Async scheduling**: vllm-ascend enables vLLM's async scheduler AND
   overlaps random-number generation on a second stream (`global_stream()`
   in their sampler). Infini does RNG on the main stream.
3. **Metadata build cost**: `InfiniAttentionMetadataBuilder` does `torch.cumsum`
   on device. On decode-only batches `cu_seqlens` has only `batch+1` entries —
   this could be built on CPU.
4. **PIECEWISE capture overhead**: our PIECEWISE mode runs attention eagerly
   between graph pieces. Each graph-piece boundary costs a stream synchronize
   and context transition. vllm-ascend appears to use a longer graph span.

## Recommended follow-ups

- **P0**: CPU profile (`py-spy record` on the throughput bench) of infini vs
  ascend to find the exact Python hotspot. Device time is known to be a
  non-issue.
- **P1**: Move `cu_seqlens` cumsum for decode to CPU, using the already-pinned
  `decode_seq_lens_cpu` that `InfiniAttentionMetadataBuilder` builds for
  `pa_d2h_free` mode. (This is in `vllm-infini/attention/metadata.py`.)
- **P1**: Test running RNG on a side stream like vllm-ascend does (may hide
  `DSARandomUniform` behind main-stream compute).
- **P2**: Expand NPUGraph capture span to eliminate per-layer host transitions
  — blocked by ATB `aclIntArray*` baking (operator-side fix).

## Warmup/capture inflation note

Total task counts under graph mode are inflated by warmup captures:

| | eager count | graph count | Ratio |
| --- | ---: | ---: | ---: |
| infini MatMulV2 | 7,081 | 24,773 | 3.5x |
| ascend MatMulV2 | 6,504 | 12,586 | 1.9x |

infini runs nearly 2x as many warmup/capture iterations as vllm-ascend. This
is pure startup cost; it does not affect steady-state throughput but does
explain why naive "total MatMulV2 time" comparisons are misleading under graph
mode.
