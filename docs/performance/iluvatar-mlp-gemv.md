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
Its final integration on current upstream still requires an Iluvatar smoke build
and tests; the PR is a draft until that check is available.

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
bias/scalar BLAS fallback. Existing `tests/test_gemm.py` covers general GEMM.
