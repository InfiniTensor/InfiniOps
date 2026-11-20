"""Correctness tests for the native CloverRecurrentGdn provider.

Golden model replicates the Triton reference: sequential GDN state update
per (request, head), L2-normalized q/k with the 1/sqrt(128) q scale, decay
from g, delta-rule state write, final-state copy to the final slot.
Cross-checked against the Triton provider, covering dead initial/final
slots, empty request ranges and head-count mapping.
"""

import infini.ops
import pytest
import torch

if not hasattr(infini.ops, "CloverRecurrentGdn"):
    pytest.skip("`CloverRecurrentGdn` is not available", allow_module_level=True)

NATIVE, TRITON = 0, 10
QSCALE = 0.08838834764831845


def golden(q, k, v, g, beta, offsets, init, fin, state):
    q, k, v, g, beta = (t.float().cpu() for t in (q, k, v, g, beta))
    _, tokens, nv, dim = v.shape
    nk = k.shape[2]
    out = torch.zeros_like(v, dtype=torch.float32)
    state_out = state.clone()
    for r in range(offsets.numel() - 1):
        begin, end = int(offsets[r]), int(offsets[r + 1])
        i0, i1 = int(init[r]), int(fin[r])
        if not (0 <= begin <= end <= tokens and 0 <= i0 < state.size(0)
                and 0 <= i1 < state.size(0)):
            continue
        for h in range(nv):
            kh = h // (nv // nk)
            s = state[i0, h].clone()
            for t in range(begin, end):
                qv = q[0, t, kh].float()
                kv = k[0, t, kh].float()
                vv = v[0, t, h].float()
                qn = qv / (qv @ qv + 1e-6).sqrt() * QSCALE
                kn = kv / (kv @ kv + 1e-6).sqrt()
                d = torch.exp(g[0, t, h])
                b = beta[0, t, h]
                s = s * d
                rem = s @ kn
                s += ((vv - rem) * b)[:, None] * kn[None, :]
                out[0, t, h] = s @ qn
            state_out[i1, h] = s
    return out, state_out


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("heads", ((4, 4), (2, 8)))  # (nk, nv)
def test_recurrent_gdn(device, dtype, heads):
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    impls = infini.ops.CloverRecurrentGdn.active_implementation_indices("nvidia")
    if NATIVE not in impls or TRITON not in impls:
        pytest.skip(f"requires impls 0 and 10, active={impls}")
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(51)
    nk, nv = heads
    tokens, dim, pool = 96, 128, 4
    # Three requests: normal, touching an empty range, one with dead slots.
    offsets = torch.tensor([0, 40, 40, 96], device=device, dtype=torch.int32)
    init = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    fin = torch.tensor([0, 3, 2], device=device, dtype=torch.int32)

    q = torch.randn(1, tokens, nk, dim, device=device, dtype=dtype) * 0.5
    k = torch.randn(1, tokens, nk, dim, device=device, dtype=dtype) * 0.5
    v = torch.randn(1, tokens, nv, dim, device=device, dtype=dtype) * 0.5
    g = torch.randn(1, tokens, nv, device=device, dtype=torch.float32) * 0.2 - 0.5
    beta = torch.rand(1, tokens, nv, device=device, dtype=torch.float32)
    state = torch.randn(pool, nv, dim, dim, device=device,
                        dtype=torch.float32) * 0.1
    out_n = torch.empty_like(v)
    out_t = torch.empty_like(v)

    infini.ops.clover_recurrent_gdn(q, k, v, g, beta, offsets, init, fin,
                                    state, out_n, stream=stream,
                                    implementation_index=NATIVE)
    infini.ops.clover_recurrent_gdn(q, k, v, g, beta, offsets, init, fin,
                                    state, out_t, stream=stream,
                                    implementation_index=TRITON)
    torch.cuda.synchronize()

    ref_out, ref_state = golden(q, k, v, g, beta, offsets.cpu(), init.cpu(),
                                fin.cpu(), state.cpu())
    scale = max(ref_out.abs().max().item(), 1.0)
    torch.testing.assert_close(out_n.float(), ref_out.to(v.device),
                               rtol=3e-2, atol=scale * 3e-2)
    torch.testing.assert_close(out_n.float(), out_t.float(), rtol=3e-2,
                               atol=scale * 3e-2)


def test_recurrent_gdn_dead_slots(device):
    """All requests have invalid initial slots: no output or state writes."""
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    impls = infini.ops.CloverRecurrentGdn.active_implementation_indices("nvidia")
    if NATIVE not in impls or TRITON not in impls:
        pytest.skip(f"requires impls 0 and 10, active={impls}")
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(52)
    dtype, nk, nv = torch.bfloat16, 2, 2
    tokens, dim, pool = 16, 128, 2
    offsets = torch.tensor([0, 16], device=device, dtype=torch.int32)
    init = torch.tensor([-1], device=device, dtype=torch.int32)
    fin = torch.tensor([0], device=device, dtype=torch.int32)
    q = torch.randn(1, tokens, nk, dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, nv, dim, device=device, dtype=dtype)
    g = torch.randn(1, tokens, nv, device=device, dtype=torch.float32) * 0.1
    beta = torch.rand(1, tokens, nv, device=device, dtype=torch.float32)
    state = torch.randn(pool, nv, dim, dim, device=device,
                        dtype=torch.float32) * 0.1
    state0 = state.clone()
    out = torch.full_like(v, float("nan"))
    infini.ops.clover_recurrent_gdn(q, k, v, g, beta, offsets, init, fin,
                                    state, out, stream=stream,
                                    implementation_index=NATIVE)
    torch.cuda.synchronize()
    # Both providers leave the output untouched for dead initial slots.
    assert torch.isnan(out.float()).all(), "dead slot must not write output"
    assert torch.equal(state, state0)
