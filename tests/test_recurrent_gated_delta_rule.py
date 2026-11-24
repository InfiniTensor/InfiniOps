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


# Packed form: `[1, T, H, D]` requests over an FP32 state pool. The reference
# runs the recurrence in float64 and applies the state-slot contract of the
# base class: the initial slot is only read, the final state goes to the final
# slot, an unusable slot zeroes the output of the request, and a malformed
# token range changes nothing. The batched overload also accepts tensors in
# its optional positions, so the packed form is called with keyword arguments.

_PACKED_NATIVE, _PACKED_TRITON = 0, 10
_PACKED_QUERY_SCALE = 0.08838834764831845


def _packed_reference(
    q, k, v, g, beta, offsets, initial_slots, final_slots, state, norm
):
    """Returns the output and the state pool after the call, in float64."""
    q, k, v, g, beta = (t.double().cpu() for t in (q, k, v, g, beta))
    _, tokens, num_v_heads, _ = v.shape
    num_k_heads = k.shape[2]
    out = torch.full(v.shape, float("nan"), dtype=torch.float64)
    state = state.double().cpu().clone()
    pool = state.shape[0]
    for request in range(offsets.numel() - 1):
        begin, end = int(offsets[request]), int(offsets[request + 1])
        initial, final = int(initial_slots[request]), int(final_slots[request])
        if not 0 <= begin <= end <= tokens:
            continue
        if not (0 <= initial < pool and 0 <= final < pool):
            out[0, begin:end] = 0
            continue
        for head in range(num_v_heads):
            key_head = head // (num_v_heads // num_k_heads)
            s = state[initial, head].clone()
            for token in range(begin, end):
                qv = q[0, token, key_head]
                kv = k[0, token, key_head]
                if norm:
                    qv = qv / (qv @ qv + 1e-6).sqrt()
                    kv = kv / (kv @ kv + 1e-6).sqrt()
                qv = qv * _PACKED_QUERY_SCALE
                s = s * torch.exp(g[0, token, head])
                delta = (v[0, token, head] - s @ kv) * beta[0, token, head]
                s = s + delta[:, None] * kv[None, :]
                out[0, token, head] = s @ qv
            state[final, head] = s
    return out, state


def _packed_implementations(device):
    if device != "cuda":
        pytest.skip("requires a CUDA-like backend")
    impls = infini.ops.RecurrentGatedDeltaRule.active_implementation_indices(device)
    if _PACKED_NATIVE not in impls and _PACKED_TRITON not in impls:
        pytest.skip(f"requires a native or Triton packed provider, active={impls}")
    return [impl for impl in (_PACKED_NATIVE, _PACKED_TRITON) if impl in impls]


def _packed_case(device, dtype, heads, index_dtype, strided):
    """Three requests: an in-place update, an empty fork and a fork of the
    same initial slot."""
    num_k_heads, num_v_heads = heads
    tokens, dim, pool = 96, 128, 5
    q = torch.randn(1, tokens, num_k_heads, dim, device=device, dtype=dtype) * 0.5
    # ‖k‖ ≈ 1 keeps the unnormalized recurrence stable over many tokens.
    k = torch.randn_like(q) * 0.09
    v = torch.randn(1, tokens, num_v_heads, dim, device=device, dtype=dtype) * 0.5
    g = torch.randn(1, tokens, num_v_heads, device=device) * 0.2 - 0.5
    beta = torch.rand(1, tokens, num_v_heads, device=device)
    if strided:
        # Columns of a wider buffer, as the narrowed projections of a model.
        wide = torch.empty(1, tokens, num_k_heads, 2 * dim, device=device, dtype=dtype)
        wide[..., :dim].copy_(q)
        wide[..., dim:].copy_(k)
        q, k = wide[..., :dim], wide[..., dim:]
    # Offsets and slots of different widths exercise both index loads.
    slot_dtype = torch.int64 if index_dtype == torch.int32 else torch.int32
    offsets = torch.tensor([0, 40, 40, tokens], device=device, dtype=index_dtype)
    initial = torch.tensor([0, 1, 1], device=device, dtype=slot_dtype)
    final = torch.tensor([0, 3, 2], device=device, dtype=slot_dtype)
    state = torch.randn(pool, num_v_heads, dim, dim, device=device) * 0.1
    return q, k, v, g, beta, offsets, initial, final, state


def _packed_run(implementation, args, out, norm):
    q, k, v, g, beta, offsets, initial, final, state = args
    infini.ops.recurrent_gated_delta_rule(
        q, k, v, g, beta, state, cu_seqlens=offsets, initial_slots=initial,
        final_slots=final, out=out, use_qk_l2norm=norm,
        stream=torch.cuda.current_stream().cuda_stream,
        implementation_index=implementation,
    )
    torch.cuda.synchronize()


def _packed_check_output(out, ref_out):
    """The data dtype rounds the output; the tolerance follows its scale."""
    scale = max(ref_out.abs().max().item(), 1.0)
    torch.testing.assert_close(
        out.double(), ref_out.to(out.device), rtol=2e-2, atol=scale * 3e-2
    )


@pytest.mark.parametrize("norm", (True, False))
@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("heads", ((4, 4), (2, 8)))  # (num_k_heads, num_v_heads)
@pytest.mark.parametrize("strided", (False, True))
def test_packed_reference_and_state(device, dtype, heads, index_dtype, norm, strided):
    torch.manual_seed(51)
    args = _packed_case(device, dtype, heads, index_dtype, strided)
    ref_out, ref_state = _packed_reference(*args, norm)
    for implementation in _packed_implementations(device):
        state = args[8].clone()
        out = torch.empty_like(args[2])
        _packed_run(implementation, args[:8] + (state,), out, norm)
        _packed_check_output(out, ref_out)
        torch.testing.assert_close(
            state.double(), ref_state.to(device), rtol=1e-3, atol=1e-3
        )
        # Slot 1 is only read: request 1 is empty and request 2 forks from it.
        assert torch.equal(state[1], args[8][1])


@pytest.mark.parametrize("broken", ("initial", "final"))
def test_packed_unusable_slot_zeroes_output(device, broken):
    """An out-of-range slot zeroes the request's output and leaves the pool alone."""
    torch.manual_seed(52)
    args = list(_packed_case(device, torch.bfloat16, (2, 2), torch.int32, False))
    slot = 6 if broken == "initial" else 7
    args[slot] = args[slot].clone()
    args[slot][2] = 99
    args = tuple(args)
    ref_out, ref_state = _packed_reference(*args, True)
    assert torch.equal(ref_state[2].to(device), args[8][2]), "no state may be written"
    for implementation in _packed_implementations(device):
        state = args[8].clone()
        out = torch.full_like(args[2], float("nan"))
        _packed_run(implementation, args[:8] + (state,), out, True)
        assert torch.equal(
            out[0, 40:], torch.zeros_like(out[0, 40:])
        ), "output must be zeroed"
        _packed_check_output(out[:, :40], ref_out[:, :40].to(device))
        assert torch.equal(state[2], args[8][2])
        assert torch.equal(state[1], args[8][1])


def test_packed_malformed_range_is_untouched(device):
    torch.manual_seed(53)
    args = _packed_case(device, torch.bfloat16, (2, 2), torch.int32, False)
    offsets = args[5].clone()
    offsets[-1] += 8  # the last request reads past the packed sequence
    args = args[:5] + (offsets,) + args[6:]
    ref_out, ref_state = _packed_reference(*args, True)
    assert torch.isnan(ref_out[0, 40:]).all()
    # The empty request still copies its initial state to its final slot.
    assert torch.equal(ref_state[3].to(device), args[8][1])
    for implementation in _packed_implementations(device):
        state = args[8].clone()
        out = torch.full_like(args[2], float("nan"))
        _packed_run(implementation, args[:8] + (state,), out, True)
        assert torch.isnan(out[0, 40:].float()).all(), "malformed range must not write"
        _packed_check_output(out[:, :40], ref_out[:, :40].to(device))
        assert torch.equal(state[3], args[8][1])
        assert torch.equal(state[1:3], args[8][1:3])
        assert torch.equal(state[4], args[8][4])


def test_packed_rejects_other_head_dims(device):
    """Both packed providers implement a head dim of 128 only."""
    torch.manual_seed(54)
    dim = 64
    q = torch.randn(1, 8, 2, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, 8, 4, dim, device=device, dtype=torch.bfloat16)
    g = torch.randn(1, 8, 4, device=device)
    beta = torch.rand_like(g)
    state = torch.randn(2, 4, dim, dim, device=device)
    offsets = torch.tensor([0, 8], device=device, dtype=torch.int32)
    initial = torch.tensor([0], device=device, dtype=torch.int32)
    final = torch.tensor([1], device=device, dtype=torch.int32)
    out = torch.empty_like(v)
    for implementation in _packed_implementations(device):
        with pytest.raises(ValueError, match="head dim"):
            infini.ops.recurrent_gated_delta_rule(
                q, k, v, g, beta, state, cu_seqlens=offsets,
                initial_slots=initial, final_slots=final, out=out,
                use_qk_l2norm=True,
                stream=torch.cuda.current_stream().cuda_stream,
                implementation_index=implementation,
            )
