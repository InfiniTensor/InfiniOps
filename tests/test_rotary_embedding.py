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


@pytest.mark.parametrize("tokens", (1, 17, 2048))
@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("is_neox", (False, True))
@pytest.mark.parametrize("inverse", (False, True))
def test_rotary_embedding_paired_mha(
    tokens, packed, is_neox, inverse, implementation_index, device
):
    if torch.device(device).type != "cuda":
        pytest.skip("the head-parallel regression targets CUDA-compatible providers")

    dtype = torch.bfloat16
    storage = torch.randn((tokens, 3, 28, 128), dtype=dtype, device=device)
    query = storage[:, 0] if packed else storage[:, 0].contiguous()
    key = storage[:, 1] if packed else storage[:, 1].contiguous()
    original_value = storage[:, 2].clone()
    positions = torch.arange(tokens, dtype=torch.int64, device=device)
    cache = torch.randn(tokens, 128, dtype=dtype, device=device)
    expected_query, expected_key = query.clone(), key.clone()
    _torch_rotary_embedding(
        positions, expected_query, expected_key, cache, 128, is_neox, 0, inverse
    )
    infini.ops.rotary_embedding(
        positions,
        query,
        key,
        cache,
        128,
        is_neox,
        0,
        inverse,
        implementation_index=implementation_index,
        stream=get_stream(query.device),
    )
    torch.testing.assert_close(query, expected_query, rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(key, expected_key, rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(storage[:, 2], original_value, rtol=0, atol=0)


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
    num_heads, num_kv_heads = 4, 2
    is_ascend = torch.device(device).type == "npu"
    if is_ascend and (rope_dim_offset != 0 or inverse):
        pytest.skip("Ascend supports full-dimension forward RoPE")
    head_size, rot_dim = (8, 8) if is_ascend else (12, 8)
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
        cos_sin_cache,
        head_size,
        is_neox,
        rope_dim_offset,
        inverse,
    )
    args = (
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
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


@pytest.mark.parametrize("has_key", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 2e-2, 1e-2),
    ),
)
def test_rotary_embedding_ascend_partial_prefix(
    has_key,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    if torch.device(device).type != "npu":
        pytest.skip("The partial-prefix regression targets Ascend")

    num_heads, num_kv_heads = 4, 2
    head_size, rot_dim = 128, 64
    positions = torch.tensor((9,), dtype=torch.int64, device=device)
    query = torch.randn(1, num_heads * head_size, dtype=dtype, device=device)
    key = (
        torch.randn(1, num_kv_heads * head_size, dtype=dtype, device=device)
        if has_key
        else None
    )
    cos_sin_cache = torch.randn(16, rot_dim, dtype=dtype, device=device)
    original_query = query.clone()
    original_key = key.clone() if key is not None else None
    expected_query = query.clone()
    expected_key = key.clone() if key is not None else None

    _torch_rotary_embedding(
        positions,
        expected_query,
        expected_key,
        cos_sin_cache,
        head_size,
        False,
        0,
        False,
    )
    result = infini.ops.rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        False,
        implementation_index=implementation_index,
        stream=get_stream(query.device),
    )

    assert result is None
    torch.testing.assert_close(query, expected_query, rtol=rtol, atol=atol)
    query_view = query.view(1, num_heads, head_size)
    original_query_view = original_query.view(1, num_heads, head_size)
    assert torch.equal(query_view[..., rot_dim:], original_query_view[..., rot_dim:])
    if key is not None:
        torch.testing.assert_close(key, expected_key, rtol=rtol, atol=atol)
        key_view = key.view(1, num_kv_heads, head_size)
        original_key_view = original_key.view(1, num_kv_heads, head_size)
        assert torch.equal(key_view[..., rot_dim:], original_key_view[..., rot_dim:])


@pytest.mark.parametrize("has_key", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 2e-2, 1e-2),
    ),
)
def test_rotary_embedding_ascend_strided_fused_qkv_view(
    has_key,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    if torch.device(device).type != "npu":
        pytest.skip("The fused-QKV stride regression targets Ascend")

    batch_size, sequence_length = 1, 10
    num_heads, num_kv_heads = 32, 2
    head_size, rot_dim = 128, 64
    query_width = num_heads * head_size
    key_width = num_kv_heads * head_size
    fused_width = query_width + 2 * key_width

    positions = torch.arange(
        sequence_length,
        dtype=torch.int64,
        device=device,
    ).view(batch_size, sequence_length)
    fused_qkv = torch.randn(
        batch_size,
        sequence_length,
        fused_width,
        dtype=dtype,
        device=device,
    )
    query = fused_qkv[..., :query_width].view(
        batch_size,
        sequence_length,
        num_heads,
        head_size,
    )
    key = (
        fused_qkv[..., query_width : query_width + key_width].view(
            batch_size,
            sequence_length,
            num_kv_heads,
            head_size,
        )
        if has_key
        else None
    )
    cos_sin_cache = torch.randn(32, rot_dim, dtype=dtype, device=device)
    expected_query = query.clone()
    expected_key = key.clone() if key is not None else None

    assert not query.is_contiguous()
    if key is not None:
        assert not key.is_contiguous()

    _torch_rotary_embedding(
        positions,
        expected_query,
        expected_key,
        cos_sin_cache,
        head_size,
        False,
        0,
        False,
    )
    result = infini.ops.rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        False,
        implementation_index=implementation_index,
        stream=get_stream(query.device),
    )

    assert result is None
    torch.testing.assert_close(query, expected_query, rtol=rtol, atol=atol)
    if key is not None:
        torch.testing.assert_close(key, expected_key, rtol=rtol, atol=atol)


@pytest.mark.parametrize("num_tokens", (1, 2, 24))
@pytest.mark.parametrize(
    "head_size, num_heads",
    ((64, 8), (64, 9), (64, 36), (64, 40), (128, 8), (128, 9), (128, 16), (128, 32)),
)
@pytest.mark.parametrize("has_key", (False, True))
@pytest.mark.parametrize("strided", (False, True))
def test_rotary_embedding_ascend_neox_decode(
    num_tokens, head_size, num_heads, has_key, strided, implementation_index, device
):
    if torch.device(device).type != "npu" or implementation_index != 0:
        pytest.skip("The decode regression targets native Ascend RoPE")

    num_kv_heads = num_heads // 2 if num_heads % 2 == 0 else 1
    half_dim = head_size // 2
    frequencies = (
        torch.arange(256).float()[:, None]
        * (1000000 ** (-torch.arange(half_dim).float() / half_dim))[None, :]
    )
    cache_cpu = torch.cat((frequencies.cos(), frequencies.sin()), -1).half()
    cache = cache_cpu.to(device)
    # Reuse the cached operator with new positions and Q/K allocations.
    for position in (35, 36):
        positions_cpu = torch.arange(position, position + num_tokens)
        query_cpu = torch.randn(num_tokens, num_heads * head_size).half()
        key_cpu = (
            torch.randn(num_tokens, num_kv_heads * head_size).half()
            if has_key
            else None
        )
        padding = []

        def to_device(value, heads):
            if not strided:
                return value.to(device)
            parent = torch.full(
                (num_tokens, heads, 2 * head_size),
                -17,
                dtype=torch.float16,
                device=device,
            )
            view = parent[..., :head_size]
            view.copy_(value.view(num_tokens, heads, head_size))
            padding.append(parent[..., head_size:])
            return view

        query = to_device(query_cpu, num_heads)
        key = to_device(key_cpu, num_kv_heads) if has_key else None
        _torch_rotary_embedding(
            positions_cpu,
            query_cpu,
            key_cpu,
            cache_cpu,
            head_size,
            True,
            0,
            False,
        )
        infini.ops.rotary_embedding(
            positions_cpu.to(device),
            query,
            key,
            cache,
            head_size,
            True,
            implementation_index=implementation_index,
            stream=get_stream(query.device),
        )
        torch.testing.assert_close(
            query.cpu().reshape(query_cpu.shape), query_cpu, rtol=1e-3, atol=1e-3
        )
        if has_key:
            torch.testing.assert_close(
                key.cpu().reshape(key_cpu.shape), key_cpu, rtol=1e-3, atol=1e-3
            )
        for untouched in padding:
            assert torch.all(untouched.cpu() == -17)


def _torch_rotary_embedding(
    positions,
    query,
    key,
    cos_sin_cache,
    head_size,
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
