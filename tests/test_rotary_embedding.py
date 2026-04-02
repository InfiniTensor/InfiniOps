import infini.ops
import pytest
import torch

from tests.utils import get_npu_stream, randn_strided, randint_strided


def _rotary_embedding(positions, query, key, cos_sin_cache,
                      head_size, rotary_dim, is_neox_style,
                      query_out, key_out, device):
    if device == "npu":
        infini.ops.rotary_embedding(
            positions, query, key, cos_sin_cache,
            head_size, rotary_dim, is_neox_style,
            query_out, key_out,
            stream=get_npu_stream(query),
        )
    else:
        infini.ops.rotary_embedding(
            positions, query, key, cos_sin_cache,
            head_size, rotary_dim, is_neox_style,
            query_out, key_out,
        )

    return query_out, key_out


def _ref_rotary_embedding(positions, query, key, cos_sin_cache,
                          head_size, rotary_dim, is_neox_style):
    """PyTorch reference for RoPE.

    ``cos_sin_cache`` layout: ``[max_seq_len, rotary_dim]`` where the first
    ``rotary_dim // 2`` columns are cos and the rest are sin.
    """
    T = query.size(0)
    R = rotary_dim
    half_R = R // 2

    cos_sin = cos_sin_cache.float()
    cos_half = cos_sin[:, :half_R]
    sin_half = cos_sin[:, half_R:]

    def apply_rope(x):
        out = x.float().clone()

        for t in range(T):
            p = positions[t].item()
            c = cos_half[p]
            s = sin_half[p]

            if is_neox_style:
                x1 = x[t, :, :half_R].float()
                x2 = x[t, :, half_R:R].float()
                out[t, :, :half_R] = c * x1 - s * x2
                out[t, :, half_R:R] = c * x2 + s * x1
            else:
                x1 = x[t, :, 0::2].float()
                x2 = x[t, :, 1::2].float()
                out[t, :, 0::2] = c * x1 - s * x2
                out[t, :, 1::2] = c * x2 + s * x1

        return out.to(x.dtype)

    return apply_rope(query), apply_rope(key)


def _assert_close(actual, expected, rtol, atol):
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Max diff: {(actual.float() - expected.float()).abs().max().item()}"
    )


@pytest.mark.parametrize(
    "num_heads, head_size",
    (
        (32, 128),
        (8, 64),
    ),
)
@pytest.mark.parametrize("is_neox_style", (True, False))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
@pytest.mark.parametrize("device", ("npu",))
def test_rotary_embedding_full(num_heads, head_size, is_neox_style,
                               dtype, rtol, atol, device):
    """Full rotary: ``rotary_dim == head_size``."""
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    num_kv_heads = num_heads
    rotary_dim = head_size
    num_tokens = 16
    max_seq_len = 64

    positions = randint_strided(
        0, max_seq_len, (num_tokens,), None,
        dtype=torch.int64, device=device,
    )
    query = randn_strided(
        (num_tokens, num_heads, head_size), None, dtype=dtype, device=device,
    )
    key = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device,
    )
    cos_sin_cache = randn_strided(
        (max_seq_len, rotary_dim), None, dtype=dtype, device=device,
    )
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)

    q_out, k_out = _rotary_embedding(
        positions, query, key, cos_sin_cache,
        head_size, rotary_dim, is_neox_style,
        query_out, key_out, device,
    )

    ref_q, ref_k = _ref_rotary_embedding(
        positions, query, key, cos_sin_cache,
        head_size, rotary_dim, is_neox_style,
    )

    _assert_close(q_out, ref_q, rtol, atol)
    _assert_close(k_out, ref_k, rtol, atol)


@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_size, rotary_dim",
    (
        (32, 8, 128, 64),
        (16, 4, 64, 32),
    ),
)
@pytest.mark.parametrize("is_neox_style", (True,))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
@pytest.mark.parametrize("device", ("npu",))
def test_rotary_embedding_partial(num_heads, num_kv_heads, head_size, rotary_dim,
                                  is_neox_style, dtype, rtol, atol, device):
    """Partial rotary: ``rotary_dim < head_size``."""
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    num_tokens = 16
    max_seq_len = 64

    positions = randint_strided(
        0, max_seq_len, (num_tokens,), None,
        dtype=torch.int64, device=device,
    )
    query = randn_strided(
        (num_tokens, num_heads, head_size), None, dtype=dtype, device=device,
    )
    key = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device,
    )
    cos_sin_cache = randn_strided(
        (max_seq_len, rotary_dim), None, dtype=dtype, device=device,
    )
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)

    q_out, k_out = _rotary_embedding(
        positions, query, key, cos_sin_cache,
        head_size, rotary_dim, is_neox_style,
        query_out, key_out, device,
    )

    ref_q, ref_k = _ref_rotary_embedding(
        positions, query, key, cos_sin_cache,
        head_size, rotary_dim, is_neox_style,
    )

    _assert_close(q_out, ref_q, rtol, atol)
    _assert_close(k_out, ref_k, rtol, atol)
