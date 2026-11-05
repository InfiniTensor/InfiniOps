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
`1a68b261d19f0ed6f76bbabeab7a25f479f05e07`; its current-base GPU smoke build
and tests remain pending, so it is submitted as a draft.

```sh
c++ -std=c++17 -Isrc tests/host/iluvatar_causal_policy_test.cpp -o /tmp/causal-policy
/tmp/causal-policy
python -m pytest tests/test_iluvatar_causal_fp32.py tests/test_causal_softmax_infinilm.py -q
```

The focused device regression compares against the same backend's materialized
mask plus Softmax, including in-place output, rectangular masks and masked NaNs.
