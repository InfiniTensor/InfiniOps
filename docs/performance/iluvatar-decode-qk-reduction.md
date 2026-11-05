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
Current-revision device compilation and smoke tests remain pending; the draft
PR separates this integration work from the earlier hardware measurements.

```sh
c++ -std=c++17 -Isrc tests/host/iluvatar_decode_policy_test.cpp -o /tmp/decode-policy
/tmp/decode-policy
python -m pytest tests/test_iluvatar_decode_reduction.py tests/test_paged_attention_infinilm.py -q
```

The focused device checks include short and non-tile-aligned lengths, B1/B2 and
the 4096-token boundary against FP32 reference Attention. Existing general
operator tests exercise fallback layouts and metadata types.
