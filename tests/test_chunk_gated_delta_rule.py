"""Metax unit tests for the chunked-prefill gated delta rule operator.

The reference below mirrors the device kernel math: per (batch, value
head) a running state s[Dv, Dk] (one row per value dim) evolves as
    k_hat = l2norm(k)  (optional)
    q_hat = l2norm(q) * rsqrt(Dk)  (when l2norm, else rsqrt(Dk))
    decay = exp(g)
    v_new = beta * (v - decay * (s @ k_hat))
    s     = decay * s + outer(v_new, k_hat)
    out   = s @ q_hat
"""

import torch
import pytest

import infini.ops

from tests.utils import get_stream


if not hasattr(infini.ops, "chunk_gated_delta_rule"):
    pytest.skip(
        "`chunk_gated_delta_rule` is not available on this platform",
        allow_module_level=True,
    )


def _sanitize(*tensors):
    """MACA's host-to-device copy reports wrong stride metadata (the
    data itself is contiguous); the operator kernels index through the
    reported strides, so normalize every device tensor here."""
    return tuple(t.contiguous() if t.is_cuda else t for t in tensors)


def _reference(q, k, v, g, beta, initial_state, use_qk_l2norm):
    """q/k/v: [B, T, H, D]; g/beta: [B, T, H]; initial_state: [pool/B, H, Dv, Dk].

    Computed in float64 so the reference stays exact regardless of the
    session-wide TF32 matmul setting (`tests/conftest.py`) and does not
    itself become the source of float32 divergence.
    """
    B, T, Hk, Dk = q.shape
    Hv = v.shape[2]
    ratio = Hv // Hk
    state = initial_state.to(torch.float64).clone()
    out = torch.empty_like(v, dtype=torch.float64)
    final = torch.empty_like(initial_state, dtype=torch.float64)

    scale = Dk**-0.5
    for b in range(B):
        for hv in range(Hv):
            hk = hv // ratio
            s = state[b, hv].clone()
            for t in range(T):
                qr = q[b, t, hk].to(torch.float64)
                kr = k[b, t, hk].to(torch.float64)
                decay = torch.exp(g[b, t, hv].to(torch.float64))
                beta_t = beta[b, t, hv].to(torch.float64)
                if use_qk_l2norm:
                    q_hat = qr * (torch.rsqrt(qr.square().sum() + 1e-6) * scale)
                    k_hat = kr * torch.rsqrt(kr.square().sum() + 1e-6)
                else:
                    q_hat = qr * scale
                    k_hat = kr
                projected = s @ k_hat
                v_new = beta_t * (v[b, t, hv].to(torch.float64) - decay * projected)
                s = decay * s + torch.outer(v_new, k_hat)
                out[b, t, hv] = s @ q_hat
            final[b, hv] = s
    return out, final


def _make_inputs(B, T, Hk, Hv, Dk, Dv, dtype, gate_dtype, device, seed=0):
    """Inputs follow the operator's production contract: q/k are l2-normalized
    per head unless `use_qk_l2norm` is set (then the kernel normalizes them
    itself). Feeding raw gaussian k would make the state recursion expand by
    a factor ~|k|^2 per token, which is numerically unstable for any
    float32 comparison. Scale keeps the state magnitude in a testable range.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, T, Hk, Dk, generator=gen)
    k = torch.randn(B, T, Hk, Dk, generator=gen)
    v = torch.randn(B, T, Hv, Dv, generator=gen)
    g = -torch.rand(B, T, Hv, generator=gen) * 0.1
    beta = torch.sigmoid(torch.randn(B, T, Hv, generator=gen))
    initial = torch.randn(B, Hv, Dv, Dk, generator=gen)
    q = q / q.square().sum(-1, keepdim=True).sqrt()
    k = k / k.square().sum(-1, keepdim=True).sqrt()
    initial = initial * 0.5
    return (
        q.to(dtype).to(device).contiguous(),
        k.to(dtype).to(device).contiguous(),
        v.to(dtype).to(device).contiguous(),
        g.to(gate_dtype).to(device).contiguous(),
        beta.to(gate_dtype).to(device).contiguous(),
        initial.to(dtype).to(device).contiguous(),
    )


@pytest.mark.parametrize("use_qk_l2norm", (False, True))
@pytest.mark.parametrize(
    "dtype, gate_dtype, rtol, atol",
    (
        # The kernel reduces the state matmuls in a different order than the
        # sequential reference; float32 recursion accumulates that rounding
        # difference, so the tolerance is a few ulps above the fp32 noise
        # floor rather than the tight per-op default.
        (torch.float32, torch.float32, 1e-4, 1e-4),
        (torch.float16, torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, torch.bfloat16, 4e-2, 4e-2),
    ),
)
def test_chunk_gated_delta_rule(dtype, gate_dtype, use_qk_l2norm, rtol, atol, device):
    B, T, Hk, Hv, Dk, Dv = 2, 7, 1, 2, 64, 64
    q, k, v, g, beta, initial = _make_inputs(
        B, T, Hk, Hv, Dk, Dv, dtype, gate_dtype, device
    )
    out = torch.empty(B, T, Hv, Dv, dtype=dtype, device=device)
    final = torch.empty(B, Hv, Dv, Dk, dtype=dtype, device=device)

    infini.ops.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial,
        out=out,
        final_state=final,
        use_qk_l2norm=use_qk_l2norm,
        chunk_size=64,
        stream=get_stream(device),
    )

    ref_out, ref_final = _reference(q, k, v, g, beta, initial, use_qk_l2norm)
    torch.testing.assert_close(out, ref_out.to(dtype), rtol=rtol, atol=atol)
    torch.testing.assert_close(final, ref_final.to(dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "dtype, gate_dtype, rtol, atol",
    (
        (torch.float32, torch.float32, 1e-4, 1e-4),
        (torch.bfloat16, torch.bfloat16, 4e-2, 4e-2),
    ),
)
def test_chunk_gated_delta_rule_state_pool(dtype, gate_dtype, rtol, atol, device):
    """Indexed state pool: two sequences read distinct slots and write
    their final states to other slots; untouched slots keep their values."""
    B, T, Hk, Hv, Dk, Dv = 2, 5, 1, 1, 64, 64
    q, k, v, g, beta, _ = _make_inputs(
        B, T, Hk, Hv, Dk, Dv, dtype, gate_dtype, device, seed=3
    )
    pool = 4
    state_pool = torch.randn(pool, Hv, Dv, Dk).to(dtype).to(device)
    initial_indices = torch.tensor((1, 3), dtype=torch.int32, device=device)
    final_indices = torch.tensor((2, 0), dtype=torch.int32, device=device)
    out = torch.empty(B, T, Hv, Dv, dtype=dtype, device=device)
    before = state_pool.clone()

    infini.ops.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        state_pool,
        initial_state_indices=initial_indices,
        final_state_indices=final_indices,
        out=out,
        use_qk_l2norm=False,
        chunk_size=64,
        stream=get_stream(device),
    )

    ref_out, _ = _reference(
        q, k, v, g, beta, state_pool[initial_indices.long()].clone(), False
    )
    torch.testing.assert_close(out, ref_out.to(dtype), rtol=rtol, atol=atol)
    ref0, final0 = _reference(
        q[0:1],
        k[0:1],
        v[0:1],
        g[0:1],
        beta[0:1],
        state_pool[1:2].clone(),
        False,
    )
    ref1, final1 = _reference(
        q[1:2],
        k[1:2],
        v[1:2],
        g[1:2],
        beta[1:2],
        state_pool[3:4].clone(),
        False,
    )
    # Seq0 read slot 1 and wrote its final state to slot 2.
    torch.testing.assert_close(state_pool[2], final0[0].to(dtype), rtol=rtol, atol=atol)
    # Seq1 read slot 3 and wrote its final state to slot 0.
    torch.testing.assert_close(state_pool[0], final1[0].to(dtype), rtol=rtol, atol=atol)
    # Untouched slots 1 and 3 keep their original values.
    torch.testing.assert_close(state_pool[1], before[1])
    torch.testing.assert_close(state_pool[3], before[3])
