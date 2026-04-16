# Sampler waste investigation — not an issue in steady state

## Summary

My earlier "27 ms of wasted greedy-sampler work per 8-prompt run" claim was
based on a misreading of msprof counts. The `Sort`, `DSARandomUniform`, and
big `Cumsum` ops only fire during graph-capture warmup and prefill, **not** in
steady-state decode. They do not show up on the decode-filtered slice of the
same profile. No optimization is warranted.

## Evidence

Raw msprof entries (3B eager, 8 prompts, 32 output tokens):

| OP | Count | Shape (input) | Avg dur (us) |
| --- | ---: | --- | ---: |
| Sort              | 2 | `[256, 151936]`  | 4633 |
| Cumsum (big)      | 2 | `[256, 151936]`  | 4932 |
| DSARandomUniform  | 2 | N/A              | 3658 |
| DSARandomUniform  | 2 | N/A              |   35 |
| Cumsum (small)    | 4 | `[1;1]`, `[7;1]` |  200 |

The big Sort/Cumsum/RNG are shape `[256, 151936]` (batch × vocab size) — this
is the sampler doing a full-vocab pass over a batch of **256** (vLLM's
graph-capture dummy batch), not our actual 8-prompt test. It fires twice per
script run (once per warmup + profile iteration) ≈ 9 ms each ≈ 18 ms total.
**This is one-shot warmup cost, not per-step.**

The small `Cumsum` entries (4 calls, 200–500 us) are the per-prefill
`cu_seqlens` build. Prefills are rare under decode-heavy workloads, so their
total impact is also bounded.

## Confirmation via decode-only slice

Running `vllm-infini/tests/decode_steady_state.py` on the same CSV:

```
=== infini-eager decode-only ops (batch=8) ===
  Decode time: 501.5 ms  (of 925.4 ms total, 54%)

  OP Type                                         Count  Total(ms)      %  Avg(us)
  ... (no Sort, no DSARandomUniform, no big Cumsum) ...
```

When the filter is `first input dim == 8` (decode batch size), none of the
sampler-waste ops appear. They are explicitly not on the decode hot path.

## Closing

Task #7 closed — no action needed on sampler. The dominant gap is host-side
(see `graph_mode_root_cause_2026-04-17.md`).

## Residual thought (low priority)

vLLM's graph capture dummy batch of 256 still does a full 256-row softmax +
sort on first startup. That adds ~18 ms of startup cost per process on
`vllm-infini`. `vllm-ascend` avoids Sort entirely by using the C++ kernel
`torch.ops._C_ascend.npu_apply_top_k_top_p` — so the same dummy-batch warmup
on ascend does not spend that time. This is pure startup, not throughput
relevant, but switching `vllm-infini/vllm_infini/sample/sampler.py`'s
`_apply_top_k_top_p` to a fused `infini.ops` kernel (if one exists) would
eliminate it. Not a current priority.
