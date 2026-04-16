import infini.ops
import pytest
import torch

from tests.utils import get_npu_stream, randn_strided, randint_strided


def _expand_cos_sin(cos_sin_cache, positions, head_size):
    """Split, neox-expand, and gather cos/sin from ``cos_sin_cache``.

    Replicates the internal gather logic of the ``RotaryEmbedding`` operator
    so that the result can be fed directly to ``ApplyRotaryPosEmb``.

    Returns:
        (cos, sin) — each ``[T, head_size]``, neox-expanded.
    """
    half_D = head_size // 2
    cos_raw = cos_sin_cache[:, :half_D]
    sin_raw = cos_sin_cache[:, half_D:]

    # Neox expansion: duplicate halves.
    cos_full = torch.cat([cos_raw, cos_raw], dim=-1)
    sin_full = torch.cat([sin_raw, sin_raw], dim=-1)

    return cos_full[positions], sin_full[positions]


def _ref_apply_rotary_pos_emb(
    query,
    key,
    cos,
    sin,
    head_size,
    is_neox_style,
):
    """PyTorch reference for apply-only RoPE with pre-gathered cos/sin."""
    T = query.size(0)
    half_D = head_size // 2

    q3d = query.view(T, -1, head_size).float()
    k3d = key.view(T, -1, head_size).float()
    cos_f = cos.float()
    sin_f = sin.float()

    def apply_rope(x):
        out = x.clone()

        for t in range(T):
            c = cos_f[t, :half_D]
            s = sin_f[t, :half_D]

            if is_neox_style:
                x1 = x[t, :, :half_D]
                x2 = x[t, :, half_D:]
                out[t, :, :half_D] = c * x1 - s * x2
                out[t, :, half_D:] = c * x2 + s * x1
            else:
                x1 = x[t, :, 0::2]
                x2 = x[t, :, 1::2]
                out[t, :, 0::2] = c * x1 - s * x2
                out[t, :, 1::2] = c * x2 + s * x1

        return out

    ref_q = apply_rope(q3d).to(query.dtype).view_as(query)
    ref_k = apply_rope(k3d).to(key.dtype).view_as(key)

    return ref_q, ref_k


def _assert_close(actual, expected, rtol, atol):
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Max diff: {(actual.float() - expected.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("num_tokens", (1, 4, 16))
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_size",
    (
        (32, 8, 128),
        (8, 8, 64),
    ),
)
@pytest.mark.parametrize("implementation_index", (0, 1))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    ((torch.float16, 1e-3, 0.01),),
)
@pytest.mark.parametrize("device", ("npu",))
def test_apply_rotary_pos_emb(
    num_tokens,
    num_heads,
    num_kv_heads,
    head_size,
    implementation_index,
    dtype,
    rtol,
    atol,
    device,
):
    """Apply-only RoPE with pre-gathered cos/sin, both CANN and ATB paths."""
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    active_indices = infini.ops.ApplyRotaryPosEmb.active_implementation_indices(device)

    if implementation_index not in active_indices:
        pytest.skip(
            f"Implementation index={implementation_index} not active on this build"
        )

    max_seq_len = 64

    positions = randint_strided(
        0,
        max_seq_len,
        (num_tokens,),
        None,
        dtype=torch.int64,
        device=device,
    )
    cos_sin_cache = randn_strided(
        (max_seq_len, head_size),
        None,
        dtype=dtype,
        device=device,
    )

    cos, sin = _expand_cos_sin(cos_sin_cache, positions, head_size)

    # 2D layout: [T, N*D] (vLLM convention).
    query = randn_strided(
        (num_tokens, num_heads * head_size),
        None,
        dtype=dtype,
        device=device,
    )
    key = randn_strided(
        (num_tokens, num_kv_heads * head_size),
        None,
        dtype=dtype,
        device=device,
    )
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)

    infini.ops.apply_rotary_pos_emb(
        query,
        key,
        cos,
        sin,
        head_size,
        True,
        query_out,
        key_out,
        implementation_index=implementation_index,
        stream=get_npu_stream(query),
    )

    ref_q, ref_k = _ref_apply_rotary_pos_emb(
        query,
        key,
        cos,
        sin,
        head_size,
        True,
    )

    _assert_close(query_out, ref_q, rtol, atol)
    _assert_close(key_out, ref_k, rtol, atol)


@pytest.mark.parametrize("num_tokens", (1, 4, 16))
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_size",
    (
        (32, 8, 128),
        (8, 8, 64),
    ),
)
@pytest.mark.parametrize("implementation_index", (0, 1))
@pytest.mark.parametrize("device", ("npu",))
def test_apply_vs_rotary_embedding(
    num_tokens,
    num_heads,
    num_kv_heads,
    head_size,
    implementation_index,
    device,
):
    """Verify ``apply_rotary_pos_emb`` matches ``rotary_embedding`` exactly."""
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    active_rope = infini.ops.RotaryEmbedding.active_implementation_indices(device)
    active_apply = infini.ops.ApplyRotaryPosEmb.active_implementation_indices(device)

    if (
        implementation_index not in active_rope
        or implementation_index not in active_apply
    ):
        pytest.skip(
            f"Implementation index={implementation_index} not active on this build"
        )

    dtype = torch.float16
    max_seq_len = 64

    positions = randint_strided(
        0,
        max_seq_len,
        (num_tokens,),
        None,
        dtype=torch.int64,
        device=device,
    )
    cos_sin_cache = randn_strided(
        (max_seq_len, head_size),
        None,
        dtype=dtype,
        device=device,
    )

    query = randn_strided(
        (num_tokens, num_heads * head_size),
        None,
        dtype=dtype,
        device=device,
    )
    key = randn_strided(
        (num_tokens, num_kv_heads * head_size),
        None,
        dtype=dtype,
        device=device,
    )

    stream = get_npu_stream(query)

    # Run existing rotary_embedding.
    ref_q = torch.empty_like(query)
    ref_k = torch.empty_like(key)
    infini.ops.rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        head_size,
        True,
        ref_q,
        ref_k,
        implementation_index=implementation_index,
        stream=stream,
    )

    # Run new apply_rotary_pos_emb with manually gathered cos/sin.
    cos, sin = _expand_cos_sin(cos_sin_cache, positions, head_size)
    new_q = torch.empty_like(query)
    new_k = torch.empty_like(key)
    infini.ops.apply_rotary_pos_emb(
        query,
        key,
        cos,
        sin,
        head_size,
        True,
        new_q,
        new_k,
        implementation_index=implementation_index,
        stream=stream,
    )

    _assert_close(new_q, ref_q, rtol=0, atol=0)
    _assert_close(new_k, ref_k, rtol=0, atol=0)
