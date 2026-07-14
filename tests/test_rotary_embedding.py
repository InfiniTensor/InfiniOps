import infini.ops
import pytest
import torch

from tests.utils import get_stream


_TEST_CASES = (
    ((4,), False, True, 0, False),
    ((4,), True, False, 2, False),
    ((2, 2), False, False, 0, True),
    ((2, 2), True, True, 2, True),
)


@pytest.mark.parametrize(
    "positions_shape, structured, is_neox, rope_dim_offset, inverse",
    _TEST_CASES,
)
@pytest.mark.parametrize("has_key", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 2e-2, 1e-2),
    ),
)
def test_rotary_embedding(
    positions_shape,
    structured,
    is_neox,
    rope_dim_offset,
    inverse,
    has_key,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    num_tokens = 4
    num_heads, num_kv_heads = 4, 2
    head_size, rot_dim = 12, 8
    positions = torch.tensor((0, 3, 5, 7), dtype=torch.int64, device=device).view(
        positions_shape
    )
    token_shape = positions_shape
    query_shape = (
        (*token_shape, num_heads, head_size)
        if structured
        else (*token_shape, num_heads * head_size)
    )
    key_shape = (
        (*token_shape, num_kv_heads, head_size)
        if structured
        else (*token_shape, num_kv_heads * head_size)
    )
    query = torch.randn(query_shape, dtype=dtype, device=device)
    key = torch.randn(key_shape, dtype=dtype, device=device) if has_key else None
    cos_sin_cache = torch.randn(16, rot_dim, dtype=dtype, device=device)
    expected_query = query.clone()
    expected_key = key.clone() if key is not None else None

    _torch_rotary_embedding(
        positions,
        expected_query,
        expected_key,
        head_size,
        cos_sin_cache,
        is_neox,
        rope_dim_offset,
        inverse,
    )
    args = (
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
    )
    if rope_dim_offset == 0 and not inverse:
        result = infini.ops.rotary_embedding(
            *args,
            implementation_index=implementation_index,
            stream=get_stream(query.device),
        )
    else:
        result = infini.ops.rotary_embedding(
            *args,
            rope_dim_offset,
            inverse,
            implementation_index=implementation_index,
            stream=get_stream(query.device),
        )

    assert result is None
    torch.testing.assert_close(query, expected_query, rtol=rtol, atol=atol)
    if key is not None:
        torch.testing.assert_close(key, expected_key, rtol=rtol, atol=atol)


def _torch_rotary_embedding(
    positions,
    query,
    key,
    head_size,
    cos_sin_cache,
    is_neox,
    rope_dim_offset,
    inverse,
):
    _apply_rotary(
        positions,
        query,
        head_size,
        cos_sin_cache,
        is_neox,
        rope_dim_offset,
        inverse,
    )

    if key is not None:
        _apply_rotary(
            positions,
            key,
            head_size,
            cos_sin_cache,
            is_neox,
            rope_dim_offset,
            inverse,
        )


def _apply_rotary(
    positions,
    data,
    head_size,
    cos_sin_cache,
    is_neox,
    rope_dim_offset,
    inverse,
):
    num_tokens = positions.numel()
    rot_dim = cos_sin_cache.shape[1]
    embed_dim = rot_dim // 2
    num_heads = data.numel() // num_tokens // head_size
    data_view = data.view(num_tokens, num_heads, head_size)
    cache = cos_sin_cache[positions.flatten()]
    cos = cache[:, :embed_dim].unsqueeze(1).float()
    sin = cache[:, embed_dim:].unsqueeze(1).float()

    if inverse:
        sin = -sin

    rotary = data_view[..., rope_dim_offset : rope_dim_offset + rot_dim]
    if is_neox:
        x = rotary[..., :embed_dim].float().clone()
        y = rotary[..., embed_dim:].float().clone()
        rotary[..., :embed_dim].copy_(x * cos - y * sin)
        rotary[..., embed_dim:].copy_(y * cos + x * sin)
    else:
        x = rotary[..., 0::2].float().clone()
        y = rotary[..., 1::2].float().clone()
        rotary[..., 0::2].copy_(x * cos - y * sin)
        rotary[..., 1::2].copy_(y * cos + x * sin)
