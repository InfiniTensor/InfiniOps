# Iluvatar Decode QK reduction

The Split-KV CTA path previously called a shared-memory warp reduction for each
of eight token scores. Each tree has six CTA barriers, so QK reduction costs
48 barriers per tile. This specialization stores all eight trees together and
shares each barrier, reducing that count to six. It preserves the FP32 addition
tree, CTA64, tile8, split4, online Softmax, workspace layout and combine kernel.

Dispatch is limited to Iluvatar BF16, int32 metadata, B1/B2, Q32/KV2/H128,
a single 4096-token page per sequence, the measured contiguous strides and no
ALiBi. Other calls instantiate the original implementation. The common kernel
template defaults to the original reduction; other backends never select the
new specialization. No additional launch, allocation, readback or counter is
added to the measured path.

## Validation and performance

On TG-V200/CoreX, the measured private implementation produced identical BF16
outputs and FP32 intermediate buffers for B1 and B2. Split plus combine median
times were 0.805160 → 0.481400 ms (B1, -40.2106%) and
0.807192 → 0.485665 ms (B2, -39.8328%). Seven alternating paired measurements
used eight executions per CUDA-event interval after correctness checks.

On top of Gate/Up GEMV, Down GEMV and Prefill fusion, two fixed GSM8K tasks
completed in 11.411599 → 9.662492 seconds (-15.3274%); throughput +18.1020%.
All sixteen warmup/formal requests were correct and token-identical, with
natural EOS and normal process exit. The 512-token budget was unchanged.

The later four-item combined evaluation on 100 new fixed targets achieved
82/100 → 84/100 accuracy and -32.5515% cumulative formal time, with 34
token-different targets, one correct→wrong and three wrong→correct. This is a
combined result, not an isolated Decode quality estimate. Its baseline was
InfiniLM `6e7f9537173cd33bb30539ebf8d92ab6d869660d`, with an internal disjoint
eight-example test-corpus prompt and strict last-`####` numeric scoring.

The submission factors the private kernel into the original template and binds
it to current InfiniOps `1a68b261d19f0ed6f76bbabeab7a25f479f05e07`.
Current-revision device compilation and focused GPU checks have passed as recorded below; historical measurements retain their exact baseline.

```sh
c++ -std=c++17 -Isrc tests/host/iluvatar_decode_policy_test.cpp -o /tmp/decode-policy
/tmp/decode-policy
python -m pytest tests/test_iluvatar_decode_reduction.py tests/test_paged_attention_infinilm.py -q
```

The focused device checks include short and non-tile-aligned lengths, B1/B2 and
the 4096-token boundary against FP32 reference Attention. Existing general
operator tests exercise fallback layouts and metadata types.

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
