# Iluvatar Prefill causal Softmax

For contiguous 3D BF16 inputs with 2 <= Q <= K <= 4096, fuse the causal
selection with a 1024-thread FP32 max/exp/sum. Divide the BF16-rounded
exponent by the unrounded FP32 sum, matching the existing InfiniLM materialized
mask plus Softmax order. The previous causal provider summed BF16-rounded
exponents. Both in-place and out-of-place output are supported; other layouts,
types and lengths retain the existing implementation.

This specializes the existing compatibility operator; it adds no public API.
The accompanying InfiniLM change opts into this provider through normal
execution submission, replacing Fill/Triu/Tril/Add plus Softmax. The fusion
uses no mask tensor, extra tensor readback, or host synchronization.

## Evidence and integration status

The measured private kernel produced 16,630,845 BF16 values exactly matching
the materialized FP32 Softmax reference. On TG-V200/CoreX, two fixed GSM8K
tasks with the MLP GEMV optimizations enabled took 7.633–7.646% less time
(request throughput +8.263–8.279%). Sixteen requests completed with correct
answers and natural EOS under the original 512-token budget.

In the later four-optimization combination, 100 paired targets took
749.203818 → 505.326616 cumulative formal seconds (-32.5515%); accuracy
82/100 → 84/100, 34 token-different targets, one correct→wrong (1097), three
wrong→correct (672, 1259, 1203), and no failed/truncated requests. Those
combined results do not isolate this operator's quality contribution. The
evaluation used eight disjoint test-corpus demonstrations and fixed strict
last-`####` numeric scoring; it is an internal paired evaluation.

Those measurements used InfiniLM `6e7f9537173cd33bb30539ebf8d92ab6d869660d`.
This submission ports the kernel to normal InfiniOps dispatch on
`1a68b261d19f0ed6f76bbabeab7a25f479f05e07`; its current-base CoreX build and GPU tests have passed as recorded below.

```sh
c++ -std=c++17 -Isrc tests/host/iluvatar_causal_policy_test.cpp -o /tmp/causal-policy
/tmp/causal-policy
python -m pytest tests/test_iluvatar_causal_fp32.py tests/test_causal_softmax_infinilm.py -q
```

The focused device regression compares against the same backend's materialized
mask plus Softmax, including in-place output, rectangular masks and masked NaNs.

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
