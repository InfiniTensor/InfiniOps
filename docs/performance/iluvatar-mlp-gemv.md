# Iluvatar small-batch MLP GEMV

The default Iluvatar GEMM implementation specializes two BF16 decode shapes:

| Projection | M | K | N |
| --- | --- | --- | --- |
| Combined Gate/Up | 1 or 2 | 4096 | 32768 |
| Down | 1 or 2 | 16384 | 4096 |

A and Y are row-major; B has strides `(1, K)`. The fast path requires
alpha=1, beta=0, no C/bias, no transpose, matching devices and aligned BF16-pair
loads. All other calls use the existing BLAS implementation. The kernel shares
each weight load between the two rows and keeps four FP32 accumulators per lane.
This changes the GEMM accumulation order relative to BLAS; token equality is not
an acceptance requirement.

## Validation and measurements

The measured private implementation on TG-V200/CoreX used these same kernel
shapes and arithmetic. The submission factors the two identical algorithms into
one template, removes experimental counters, and adds an alignment fallback.
Its current submitted-code CoreX build and GPU checks are recorded below.

Historical same-card measurements, using two fixed GSM8K tasks with a 512-token
greedy budget, found Gate/Up reduced application completion time by
9.22–9.28%. Adding Down to that version reduced time by a further 6.58–6.75%.
These are sequential experiments with different baselines, not additive gains.

The subsequent **four-optimization combination** (these two GEMVs, fused
Prefill causal Softmax, and Decode QK reduction) was compared against InfiniLM
`6e7f9537173cd33bb30539ebf8d92ab6d869660d` on 100 new fixed GSM8K targets.
Both versions for each target ran on the same card. The first 20 targets ran
serially on one card; the remaining 80 were sharded over two cards. The sum of
paired formal batch makespans was 749.203818 → 505.326616 seconds (-32.5515%);
request throughput increased 48.2613%. Output lengths changed, so token
throughput (+51.6653%) is reported separately.

Accuracy was 82/100 → 84/100; all requests completed without errors or
truncation. There were 34 token-different targets, including one correct→wrong
(1097: reference 480, output 480→448) and three wrong→correct
(672, 1259, 1203). These combined results do not isolate either GEMV's quality
effect. Eight demonstrations came from a disjoint subset of the same test
corpus; this is an internal paired evaluation, not the standard benchmark split.
Generation budget, EOS and strict last-`####` numeric scoring were unchanged.

InfiniLM main advanced to `cc6d52ae04f4c9ff17229bc7fc13a9ad105aca6d`
(auto-tuning initialization) after that experiment. This PR is based on InfiniOps
`1a68b261d19f0ed6f76bbabeab7a25f479f05e07`; the historical timings do not
measure this rebased revision.

## Reproduce focused checks

```sh
c++ -std=c++17 -Isrc tests/host/iluvatar_gemv_policy_test.cpp -o /tmp/gemv-policy
/tmp/gemv-policy
python -m pytest tests/test_iluvatar_mlp_gemv.py -q
```

The portable test checks eligibility/fallback boundaries. The device test covers
both shapes and batch sizes against a deterministic FP32 reference, plus the
supported no-C scalar BLAS fallback. Existing `tests/test_gemm.py` covers general GEMM.

## Current submitted-code CoreX validation

CoreX current-code validation completed on 2026-11-06 against InfiniLM
`12a4492f70dd33235e3d142e333ad4a9629d3b7b` and InfiniOps
`1a68b261d19f0ed6f76bbabeab7a25f479f05e07`. Both complete SDKs compiled
and imported. Both roles used the identical documented CoreX validation adapter
and ABI0, with automatic tuning disabled. The adapter includes existing CUDA
Argmax/Triu/Tril registrations, unchanged rejecting unavailable factories, a
copy-on-write read-only-view loader, and the existing paged-only FlashVarlen
provider's metadata gate. See the companion InfiniLM
`docs/performance/corex-validation/README.md` for exact scope.

The affected native GPU tests passed without skips: GEMV 4, causal Softmax 8,
Decode Attention 8, plus 7 common Argmax/mask checks. The submitted device tests
now explicitly bind implementation index 0. GEMV fallback uses the supported
no-C, alpha=0.5 contract; this backend rejects C/bias at its base operator.

Original tasks 554/172 ran on one TG-V200 in forward and reverse role order.
Each of the four model processes used one simultaneous two-request warmup wave
and one simultaneous two-request measured wave: 16 requests total, 8 measured.
All 16 answers were correct with natural EOS, zero errors/truncations, normal
scope return, final library maps, atexit and actual OS exit0. The original
512-token greedy budget, EOS=[2,73440], strict last-#### numeric scorer, inputs
and weights were unchanged. Two physical rows, full Prefill and graph off were
held fixed. No new independent quality questions were added.

Mean two-request measured-wave time was **14.976775177 ->
9.782192262 seconds (-34.6843%)**.
Completed request throughput was **0.133540096 -> 0.204453148
requests/s (+53.1024%)**. Published token throughput
was 20.631944884 -> 31.588011330 tokens/s and is reported separately.
The time denominator includes queue and text processing and excludes loading,
warmup and final shutdown. Both role orders reduced application time; this is
the combined four-optimization gain, not an individual-PR or component sum.

| Order | Official wave, s | Combined wave, s | Time reduction |
| --- | ---: | ---: | ---: |
| forward | 14.973374084 | 9.782256734 | 34.6690% |
| reverse | 14.980176271 | 9.782127790 | 34.6995% |

Token-different task IDs across the two measured orders: none. Answers and token differences are recorded separately.
Raw clocks, output IDs, strict scores, native receipts and ordinary exits were independently recomputed. This two-question integration check is kept separate from the historical 100-question experiment above.
