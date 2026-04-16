# E2E Throughput Baseline — vllm-infini (PieceWise) vs vllm-ascend (graph)

## Summary

| Model | vllm-infini total tok/s | vllm-ascend total tok/s | Ratio | Target 80%? |
| --- | ---: | ---: | ---: | --- |
| `Qwen2.5-0.5B-Instruct` | 7,940.2 | 15,525.2 | **51.14%** | **FAR BELOW** |
| `Qwen2.5-3B-Instruct`   | 5,299.1 | 10,147.6 | **52.22%** | **FAR BELOW** |

Mode: default (no `--enforce-eager`). On vllm-infini this is PIECEWISE (attention eager, other ops NPUGraph); on vllm-ascend this is their full-graph / ACL-graph mode.

**The graph gap is ~2x, much wider than the eager gap.** Eager is already at 70-79% — switching on graph mode on both sides puts vllm-infini further behind, because vllm-ascend extracts a ~1.5x speedup from graph mode while vllm-infini extracts essentially **0%**.

Cross-mode:

| Model | Plugin | eager tok/s | graph tok/s | Graph speedup |
| --- | --- | ---: | ---: | ---: |
| 0.5B | vllm-infini | 7,188.0 |  7,940.2 | **1.10x** |
| 0.5B | vllm-ascend | 10,150.9 | 15,525.2 | 1.53x |
| 3B   | vllm-infini | 5,290.7 |  5,299.1 | **1.00x** (no gain) |
| 3B   | vllm-ascend | 6,690.4 | 10,147.6 | 1.52x |

## Run environment

Same as `e2e_baseline_eager_2026-04-17.md`:

- Ascend 910B4 x 1 (device 1), CANN 8.5.1
- torch-npu 2.9.0.post1, vllm 0.18.0
- vllm-infini commit `7b6099f`, InfiniOps commit `a75c7f8`
- Container: `infiniops-bench-ascend-v2` (image `infiniops-ci/ascend:latest`)
- Date: 2026-04-17

## Exact commands

```bash
docker exec infiniops-bench-ascend-v2 bash -c \
  "VLLM_PLUGINS=infini python3 -m vllm.entrypoints.cli.main bench throughput \
     --model /workspace/models/Qwen/Qwen2.5-3B-Instruct \
     --dtype float16 --max-model-len 2048 \
     --dataset-name random --random-input-len 128 --random-output-len 128 \
     --num-prompts 256 \
     --output-json /tmp/bench_infini_graph_3b.json"
# Same for vllm-ascend (VLLM_PLUGINS=ascend) and for Qwen2.5-0.5B-Instruct.
# Default compilation mode is piecewise — no extra flags needed.
```

JSONs persisted in the container at `/tmp/bench_{infini,ascend}_graph_{0p5b,3b}.json`.

## Throughput matrix (graph mode)

| Model | Plugin | Elapsed (s) | req/s | Total tok/s | Output tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5B | vllm-infini | 8.255 | 31.02 |  7,940.2 | 3,970.1 |
| 0.5B | vllm-ascend | 4.221 | 60.65 | 15,525.2 | 7,762.6 |
| 3B   | vllm-infini | 12.367 | 20.70 |  5,299.1 | 2,649.6 |
| 3B   | vllm-ascend |  6.459 | 39.64 | 10,147.6 | 5,073.8 |

## Key findings

1. **vllm-infini PIECEWISE extracts almost no speedup over eager** — 1.00x on 3B, 1.10x on 0.5B.
2. **vllm-ascend's graph mode gives ~1.52x speedup** on both models.
3. The gap to the 80% target is therefore **driven almost entirely by the graph-mode gap**, not by per-op kernel cost.
4. Why PIECEWISE is underperforming (from `vllm-infini/CLAUDE.md` and prior memory):
   - Attention still runs eagerly between graph pieces (ATB/ACLNN bake per-call `aclIntArray*` at capture; pa replay produces garbage). That means ~36 attention layers x per-step host-side work per step remain.
   - Launch / dispatch overhead on Ascend is high per op, and PIECEWISE breaks the graph at every attention layer.
   - Our prior memory [Torchair profiling findings] already proved the gap is per-op decomposition (4.4x launches), not graph compilation.
5. vllm-ascend's 1.52x speedup suggests they are capturing more (or all) of the decode path as a single graph — or they eliminate far more per-step host work. Understanding their actual cudagraph_mode + how they avoid the same `aclIntArray*` bake issue is the highest-leverage investigation.

## Conclusions & recommendations

P0 (before touching kernel perf):

- **Profile vllm-ascend's graph mode with msprof** to measure its decode-step launch count vs ours. Compare the launch count + per-step CPU time; that diff, not the per-op cost, is the main PIECEWISE bottleneck.
- **Investigate `INFINI_DECODE_ATTENTION=fa` and `pa_d2h_free`** more carefully — those modes eliminate per-layer `aclrtMemcpy` D2H. Rerun this matrix with each mode and compare.
- **Investigate `INFINI_USE_TORCHAIR=1`** — torchair may capture more of the decode step end-to-end.

P1:

- Combine graph-mode improvements with the eager improvements from Task #2 (sampler greedy short-circuit, `MatMulV2` gap, `Cumsum` removal). Eager gains compound into graph mode only if the ops are actually invoked per step inside the graph.

P2 (handoff to `operator`):

- Decode-time attention cannot be graph-captured because ACLNN/ATB bake `aclIntArray*` at capture. If `operator` can expose variants that consume a device tensor for sequence lengths instead of a baked host array, graph capture becomes viable. This is the big structural lever.

Next: begin Task #4 (detailed per-op cost analysis from the msprof data already collected) and, separately, reproduce vllm-ascend's graph-mode profile to ground P0 decisions.
