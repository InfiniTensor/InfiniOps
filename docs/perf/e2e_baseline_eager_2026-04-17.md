# E2E Throughput Baseline — vllm-infini (eager) vs vllm-ascend (eager)

## Summary

| Model | vllm-infini total tok/s | vllm-ascend total tok/s | Ratio | Target 80%? |
| --- | ---: | ---: | ---: | --- |
| `Qwen2.5-0.5B-Instruct` | 7,188.0 | 10,150.9 | **70.82%** | below |
| `Qwen2.5-3B-Instruct`   | 5,290.7 |  6,690.4 | **79.08%** | ~at target |

Mode: `--enforce-eager`, dtype float16, random dataset (128 in / 128 out), 256 prompts.

Observations:

- 3B eager is essentially at the 80% target (within 1 pp).
- 0.5B eager is further behind (70.8%) — a smaller model leaves less room to hide launch/dispatch overhead, so the penalty of any extra op cost is magnified.
- msprof op breakdown (3B) shows the dominant delta is ~+50 ms on `MatMulV2` (10% more GEMM time) and ~+27 ms of infini-only overhead from `Cumsum`, `Sort`, `DSARandomUniform`, and a larger `ZerosLike` — see "Op-level diff" below.

## Run environment

| Key | Value |
| --- | --- |
| Host | Ascend 910B4 x 8 (1 NPU used: device 1) |
| Container | `infiniops-bench-ascend-v2` (image `infiniops-ci/ascend:latest`) |
| npu-smi | 25.5.1 |
| CANN | 8.5.1 (`/usr/local/Ascend/cann-8.5.1`) |
| torch-npu | 2.9.0.post1+gitee7ba04 |
| vllm | 0.18.0 (`/vllm-workspace/vllm`, empty wheel shim) |
| vllm-ascend | 0.18.0rc1 |
| vllm-infini commit | `7b6099f` — fix: revert to PIECEWISE for all decode attention modes |
| InfiniOps commit | `a75c7f8` — test(ascend): broaden rope impl/dtype coverage, add padding-slot case, narrow PA skip probe |
| Date | 2026-04-17 |

## Exact commands

Throughput (per model x plugin):

```bash
# vllm-infini eager.
docker exec infiniops-bench-ascend-v2 bash -c \
  "VLLM_PLUGINS=infini python3 -m vllm.entrypoints.cli.main bench throughput \
     --model /workspace/models/Qwen/Qwen2.5-3B-Instruct \
     --dtype float16 --max-model-len 2048 \
     --dataset-name random --random-input-len 128 --random-output-len 128 \
     --num-prompts 256 --enforce-eager \
     --output-json /tmp/bench_infini_eager_3b.json"

# vllm-ascend eager (same but VLLM_PLUGINS=ascend).
```

msprof op-level breakdown (3B, 8 prompts, 32 output tokens, eager):

```bash
docker exec infiniops-bench-ascend-v2 bash -c \
  "VLLM_PLUGINS=infini msprof --output=/tmp/prof_infini_eager_3b \
     --application=\"python3 /workspace/vllm-infini/tests/profile_compare.py \
       --model /workspace/models/Qwen/Qwen2.5-3B-Instruct \
       --num-prompts 8 --output-len 32 --enforce-eager\""
# Then run tests/parse_op_summary.py on the emitted op_summary_*.csv.
```

Full throughput JSONs live inside the container at:

- `/tmp/bench_infini_eager_0p5b.json`, `/tmp/bench_infini_eager_3b.json`
- `/tmp/bench_ascend_eager_0p5b.json`, `/tmp/bench_ascend_eager_3b.json`

## Throughput matrix

| Model | Plugin | Elapsed (s) | req/s | Total tok/s | Output tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5B | vllm-infini | 9.117 | 28.08 |  7,188.0 | 3,594.0 |
| 0.5B | vllm-ascend | 6.456 | 39.65 | 10,150.9 | 5,075.4 |
| 3B   | vllm-infini | 12.387 | 20.67 |  5,290.7 | 2,645.4 |
| 3B   | vllm-ascend |  9.796 | 26.13 |  6,690.4 | 3,345.2 |

Same-plugin workloads are processed in parallel (vLLM async scheduling enabled on both). `total tok/s` = input+output throughput; input and output are equal at 128/128.

## Op-level diff (3B eager, msprof, 8 prompts x 32 tokens)

Total device time: infini 925.4 ms, ascend 836.9 ms (+10.6%).

Top entries where infini > ascend (regression candidates):

| OP Type (infini) | infini (us) | ascend counterpart (us) | delta (us) | note |
| --- | ---: | ---: | ---: | --- |
| MatMulV2           | 473,925 | MatMulV2 423,826 | **+50,099** | +11.8% decode GEMM — suggests GEMM tiling / dtype alignment gap. |
| MatMulV3           | 209,958 | MatMulV3 209,290 | +668 | parity for prefill GEMM. |
| Cumsum             |  10,681 | *(not present)*  | **+10,681** | infini-only; likely cumsum for `cu_seqlens` built on-device. |
| Sort               |   9,267 | *(not present)*  | **+9,267**  | sampler pre-sorts probs even though `temperature=0.0` is greedy. |
| DSARandomUniform   |   7,385 | *(not present)*  | **+7,385**  | RNG in sampler; also wasted under greedy. |
| PagedAttentionMaskNdKernel | 40,059 | FusedInferAttentionScore 37,672 | +2,387 | decode attention kernel choice (ATB PA vs ACLNN FIA); roughly parity. |
| SwiGlu             |  49,543 | SwiGlu 48,223   | +1,320 | parity. |
| ZerosLike          |  27,417 | ZerosLike 29,299 | -1,882 | infini wins slightly. |
| AddRmsNorm         |  24,532 | AddRmsNormBias 26,200 | -1,668 | infini wins. |
| AtbRopeKernel      |  13,449 | _triton_rope 28,725 | **-15,276** | infini RoPE is significantly faster than ascend's triton RoPE. |

Net device-time deficit: infini ~+88 ms over the whole 8-prompt run. The 0.5B model elasticity suggests a lot of that is fixed per-op overhead, not FLOPs.

Ascend exclusives not present in infini: `BatchMatMulV2`, `Transpose`, `Range`, `DropOutDoMask`, `ScatterElementsV2`, `LinearIndex`, `Reciprocal`, `Pow`, `Exp`, `ReduceMax`, `DSAGenBitMask`, `PpMatmulAccumAtomicKernel`, `Tile` — mostly sampler / helper ops.

Infini exclusives not present in ascend: `PagedAttentionMaskNdKernel` (ATB PA), `AtbRopeKernel` (ATB RoPE), `Cumsum`, `Sort`, `DSARandomUniform`, `MaskedFill`, `SoftmaxV2`, `Less`, `Log`, `Neg`, `GreaterEqual`, `LessEqual`, `AsStrided`, `ViewCopy`, `MemSet`, `FusedInferAttentionScore` (only 144 prefill calls).

## Key findings

1. **3B eager is 79.1% — one percentage point short of the 80% target.** With a single non-trivial optimization it should clear the bar.
2. **MatMulV2 accounts for ~50% of device time in both plugins**; infini is ~12% slower on it (`+50 ms` out of 925 ms total). This is the largest single improvement target.
3. **Sampler overhead under greedy decoding is wasted on infini.** `Sort` + `DSARandomUniform` + `ArgMaxV2` + others add up to ~17 ms per 8-prompt-32-token run, while vllm-ascend runs a much leaner greedy path (no `Sort`, no `DSARandomUniform`). Under greedy sampling (`temperature=0.0`), `InfiniSampler` / `InfiniTopKTopPSampler` should short-circuit to pure `argmax`.
4. **Infini's `AtbRopeKernel` is a win** — less than half the time of ascend's `_triton_rope` (13.4 ms vs 28.7 ms). Keep it.
5. **Infini's `PagedAttentionMaskNdKernel` decode kernel is at parity with ascend's `FusedInferAttentionScore`** once call counts are equal. No action needed there for eager mode.

## Conclusions & recommendations

Actionable next steps (for Task #4 / Task #5):

- **P0** — short-circuit greedy sampling in `vllm_infini/sample/sampler.py`: when all requests have `temperature=0`, skip `Sort`, `DSARandomUniform`, and the sort/gather cutoff path. Target saving: ~17 ms / step for our 8-prompt microbench, proportionally larger for higher batch sizes.
- **P1** — investigate the ~12% `MatMulV2` slowdown. Candidates: per-call aclnn matmul cache miss, per-call `AsStrided` (144 counts) forcing non-contiguous input, dtype upcast. The `AsStrided` spike is suspicious — worth tracking to a single call site.
- **P2** — remove the infini-only `Cumsum` if it is only used to build `cu_seqlens` for a sequence-length metadata tensor that vLLM already provides on CPU.

Move to Task #3 (PieceWise throughput) next. The MatMulV2 analysis and the greedy sampler fix should then be filed as operator / plugin tasks.
