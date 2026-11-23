"""Metax unit tests for the single-token (decode) gated delta rule operator.

The reference mirrors the device kernel math. Per (batch, value head),
for the single token t=0 with state row s[Dv, Dk] (one row per value
dim):
    k_hat = l2norm(k)  (optional);  q_hat = l2norm(q) * rsqrt(Dk) or rsqrt(Dk)
    decay = exp(g);  beta_t = beta
    delta_v = (v - (decay * s_row) . k_hat) * beta_t      (per value dim)
    out     = (decay * s_row) . q_hat + delta_v * (k_hat . q_hat)
    s_row   = decay * s_row + delta_v * k_hat
"""

import torch
import pytest

import infini.ops

from tests.utils import get_stream


if not hasattr(infini.ops, "recurrent_gated_delta_rule"):
    pytest.skip(
        "`recurrent_gated_delta_rule` is not available on this platform",
        allow_module_level=True,
    )


def _sanitize(*tensors):
    """MACA's host-to-device copy reports wrong stride metadata (the
    data itself is contiguous); the operator kernels index through the
    reported strides, so normalize every device tensor here."""
    return tuple(t.contiguous() if t.is_cuda else t for t in tensors)


def _reference(q, k, v, g, beta, initial_state, use_qk_l2norm):
    """q/k/v: [B, 1, H, D]; g/beta: [B, 1, H]; initial_state: [pool/B, H, Dv, Dk].

    Computed in float64 so the reference stays exact regardless of the
    session-wide TF32 matmul setting (`tests/conftest.py`).
    """
    B, _, Hk, Dk = q.shape
    Hv = v.shape[2]
    ratio = Hv // Hk
    state = initial_state.to(torch.float64).clone()
    out = torch.empty_like(v, dtype=torch.float64)
    final = torch.empty_like(initial_state, dtype=torch.float64)

    query_scale = Dk**-0.5
    for b in range(B):
        for hv in range(Hv):
            hk = hv // ratio
            qr = q[b, 0, hk].to(torch.float64)
            kr = k[b, 0, hk].to(torch.float64)
            decay = torch.exp(g[b, 0, hv].to(torch.float64))
            beta_t = beta[b, 0, hv].to(torch.float64)
            if use_qk_l2norm:
                q_hat = qr * torch.rsqrt(qr.square().sum() + 1e-6) * query_scale
                k_hat = kr * torch.rsqrt(kr.square().sum() + 1e-6)
            else:
                q_hat = qr * query_scale
                k_hat = kr
            s = state[b, hv].to(torch.float64)  # [Dv, Dk]
            decayed = s * decay  # [Dv, Dk]
            kv = decayed @ k_hat  # [Dv]
            delta = (v[b, 0, hv].to(torch.float64) - kv) * beta_t
            out[b, 0, hv] = decayed @ q_hat + delta * (k_hat @ q_hat)
            final[b, hv] = s * decay + torch.outer(delta, k_hat)
    return out, final


def _make_inputs(B, Hk, Hv, Dk, Dv, dtype, gate_dtype, device, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, 1, Hk, Dk, generator=gen).to(dtype).to(device).contiguous()
    k = torch.randn(B, 1, Hk, Dk, generator=gen).to(dtype).to(device).contiguous()
    v = torch.randn(B, 1, Hv, Dv, generator=gen).to(dtype).to(device).contiguous()
    g = (
        (-torch.rand(B, 1, Hv, generator=gen) * 0.1)
        .to(gate_dtype)
        .to(device)
        .contiguous()
    )
    beta = (
        torch.sigmoid(torch.randn(B, 1, Hv, generator=gen))
        .to(gate_dtype)
        .to(device)
        .contiguous()
    )
    initial = (
        torch.randn(B, Hv, Dv, Dk, generator=gen).to(dtype).to(device).contiguous()
    )
    return _sanitize(q, k, v, g, beta, initial)


@pytest.mark.parametrize("use_qk_l2norm", (False, True))
@pytest.mark.parametrize(
    "dtype, gate_dtype, rtol, atol",
    (
        (torch.float32, torch.float32, 2e-5, 2e-5),
        (torch.float16, torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, torch.bfloat16, 4e-2, 4e-2),
    ),
)
def test_recurrent_gated_delta_rule(
    dtype, gate_dtype, use_qk_l2norm, rtol, atol, device
):
    B, Hk, Hv, Dk, Dv = 2, 1, 2, 64, 64
    q, k, v, g, beta, initial = _make_inputs(
        B, Hk, Hv, Dk, Dv, dtype, gate_dtype, device
    )
    out = torch.empty(B, 1, Hv, Dv, dtype=dtype, device=device)
    final = torch.empty(B, Hv, Dv, Dk, dtype=dtype, device=device)

    infini.ops.recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial,
        out=out,
        final_state=final,
        use_qk_l2norm=use_qk_l2norm,
        stream=get_stream(device),
    )

    ref_out, ref_final = _reference(q, k, v, g, beta, initial, use_qk_l2norm)
    torch.testing.assert_close(out, ref_out.to(dtype), rtol=rtol, atol=atol)
    torch.testing.assert_close(final, ref_final.to(dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "dtype, gate_dtype, rtol, atol",
    (
        (torch.float32, torch.float32, 2e-5, 2e-5),
        (torch.bfloat16, torch.bfloat16, 4e-2, 4e-2),
    ),
)
def test_recurrent_gated_delta_rule_state_pool(dtype, gate_dtype, rtol, atol, device):
    """Indexed state pool: two sequences read distinct slots and write
    their final states to other slots; untouched slots keep their values."""
    B, Hk, Hv, Dk, Dv = 2, 1, 1, 64, 64
    q, k, v, g, beta, _ = _make_inputs(
        B, Hk, Hv, Dk, Dv, dtype, gate_dtype, device, seed=5
    )
    pool = 4
    state_pool = torch.randn(pool, Hv, Dv, Dk).to(dtype).to(device)
    initial_indices = torch.tensor((1, 3), dtype=torch.int32, device=device)
    final_indices = torch.tensor((2, 0), dtype=torch.int32, device=device)
    out = torch.empty(B, 1, Hv, Dv, dtype=dtype, device=device)
    before = state_pool.clone()

    infini.ops.recurrent_gated_delta_rule(
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
        stream=get_stream(device),
    )

    ref_out, _ = _reference(
        q, k, v, g, beta, state_pool[initial_indices.long()].clone(), False
    )
    torch.testing.assert_close(out, ref_out.to(dtype), rtol=rtol, atol=atol)
    _, final0 = _reference(
        q[0:1], k[0:1], v[0:1], g[0:1], beta[0:1], state_pool[1:2].clone(), False
    )
    _, final1 = _reference(
        q[1:2], k[1:2], v[1:2], g[1:2], beta[1:2], state_pool[3:4].clone(), False
    )
    # Seq0 read slot 1 and wrote its final state to slot 2.
    torch.testing.assert_close(state_pool[2], final0[0].to(dtype), rtol=rtol, atol=atol)
    # Seq1 read slot 3 and wrote its final state to slot 0.
    torch.testing.assert_close(state_pool[0], final1[0].to(dtype), rtol=rtol, atol=atol)
    # Untouched slots 1 and 3 keep their original values.
    torch.testing.assert_close(state_pool[1], before[1])
    torch.testing.assert_close(state_pool[3], before[3])
