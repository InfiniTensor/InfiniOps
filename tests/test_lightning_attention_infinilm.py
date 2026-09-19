import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided

# (batch, seq_len, num_heads, head_dim, pool_size)
_SHAPES = (
    (1, 1, 2, 4, 1),
    (2, 3, 2, 4, 4),
    (3, 1, 4, 8, 6),
    (1, 5, 4, 8, 2),
)

# The recurrent state is stored in the tensor dtype, so rounding accumulates
# over the sequence; reduced-precision cases therefore need looser tolerances
# than a single elementwise operator would.
_DTYPE_CASES = (
    (torch.float32, 1e-5, 1e-6),
    (torch.float16, 2e-2, 2e-2),
    (torch.bfloat16, 4e-2, 4e-2),
)


def _make_tensors(shape, dtype, device):
    batch, seq_len, num_heads, head_dim, pool_size = shape

    q = randn_strided((batch, seq_len, num_heads, head_dim), None, dtype=dtype, device=device)
    k = randn_strided((batch, seq_len, num_heads, head_dim), None, dtype=dtype, device=device)
    v = randn_strided((batch, seq_len, num_heads, head_dim), None, dtype=dtype, device=device)
    # The decay has to be positive, otherwise `exp(-slope)` would grow.
    slope = randn_strided((num_heads,), None, dtype=torch.float32, device=device).abs() * 0.5
    state = randn_strided(
        (pool_size, num_heads, head_dim, head_dim), None, dtype=dtype, device=device
    )

    # Deliberately read and write different pool rows so that the test also
    # covers the "initial row must stay untouched" contract. Rows are disjoint
    # across requests because requests may execute concurrently.
    initial_indices = torch.arange(batch, dtype=torch.int32, device=device) % pool_size
    final_indices = (initial_indices + batch) % pool_size

    out = empty_strided((batch, seq_len, num_heads, head_dim), None, dtype=dtype, device=device)

    return q, k, v, slope, state, initial_indices, final_indices, out


def _torch_lightning_attention(q, k, v, slope, state, initial_indices, final_indices):
    """Recurrent reference: S <- ratio * S + k^T v ; out = q @ S."""

    q = q.float()
    k = k.float()
    v = v.float()
    state = state.float().clone()
    out = torch.empty_like(q)
    ratio = torch.exp(-slope.float())

    for b in range(q.shape[0]):
        initial_row = int(initial_indices[b].item())
        final_row = int(final_indices[b].item())

        current = state[initial_row].clone()  # [heads, head_dim, head_dim]
        for t in range(q.shape[1]):
            current = ratio[:, None, None] * current + k[b, t].unsqueeze(-1) * v[b, t].unsqueeze(-2)
            # The implementation stores the recurrent state in the tensor dtype,
            # so the reference has to round it before the next read.
            current = current.to(state.dtype)
            out[b, t] = torch.einsum("hd,hde->he", q[b, t], current.float())

        state[final_row] = current

    return out, state


def _run_lightning_attention(q, k, v, slope, state, initial_indices, final_indices, out):
    infini.ops.lightning_attention_infinilm(
        q,
        k,
        v,
        slope,
        state,
        initial_indices,
        final_indices,
        out,
        stream=get_stream(q.device),
    )
    return out


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPE_CASES)
def test_lightning_attention_infinilm(shape, dtype, device, rtol, atol):
    tensors = _make_tensors(shape, dtype, device)

    return Payload(_run_lightning_attention, _reference_out, tensors, {}, rtol=rtol, atol=atol)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPE_CASES)
def test_lightning_attention_infinilm_state_pool(shape, dtype, device, rtol, atol):
    tensors = _make_tensors(shape, dtype, device)

    return Payload(_run_lightning_attention_state, _reference_state, tensors, {}, rtol=rtol, atol=atol)


def _reference_out(q, k, v, slope, state, initial_indices, final_indices, out):
    reference, _ = _torch_lightning_attention(q, k, v, slope, state, initial_indices, final_indices)
    out.copy_(reference.to(out.dtype))
    return out


def _reference_state(q, k, v, slope, state, initial_indices, final_indices, out):
    _, reference = _torch_lightning_attention(q, k, v, slope, state, initial_indices, final_indices)
    state.copy_(reference.to(state.dtype))
    return state


def _run_lightning_attention_state(q, k, v, slope, state, initial_indices, final_indices, out):
    _run_lightning_attention(q, k, v, slope, state, initial_indices, final_indices, out)
    return state
