import math

import infini.ops
import pytest
import torch

from tests.test_flash_attn_varlen_func import _reference_varlen_attention


if not hasattr(infini.ops, "FlashAttnWithKvcache"):
    pytest.skip(
        "`FlashAttnWithKvcache` is not available on this platform",
        allow_module_level=True,
    )


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 2e-2, 2e-2),
    ),
)
@pytest.mark.parametrize("num_kv_heads", (2, 4))
@pytest.mark.parametrize("implementation_index", (8,))
def test_moore_paged_flash_attn_with_kvcache(
    device, implementation_index, num_kv_heads, head_dim, dtype, rtol, atol
):
    if device != "musa":
        pytest.skip("paged Moore decode requires the Moore backend")

    cache_seqlens = (1, 255, 256, 257)
    batch_size = len(cache_seqlens)
    num_heads = 4
    page_size = 256
    q = torch.randn(
        (batch_size, 1, num_heads, head_dim + 5), dtype=dtype, device=device
    )[..., :head_dim]
    k_cache = torch.randn(
        (5, page_size, num_kv_heads, head_dim + 7), dtype=dtype, device=device
    )[..., :head_dim]
    v_cache = torch.randn(
        (5, page_size, num_kv_heads, head_dim + 11), dtype=dtype, device=device
    )[..., :head_dim]
    block_table = torch.tensor(
        ((3, -1), (1, -1), (4, -1), (2, 0)),
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens_tensor = torch.tensor(cache_seqlens, dtype=torch.int32, device=device)
    alibi_slopes = torch.linspace(
        0.01, 0.04, num_heads, dtype=torch.float32, device=device
    )
    out = torch.full(
        (batch_size, 1, num_heads, head_dim + 13),
        math.nan,
        dtype=dtype,
        device=device,
    )[..., :head_dim]
    k_before = k_cache.clone()
    v_before = v_cache.clone()

    expected = _reference_varlen_attention(
        q[:, 0],
        k_cache,
        v_cache,
        (1,) * batch_size,
        cache_seqlens,
        0.125,
        True,
        (-1, -1),
        block_table,
        alibi_slopes,
    ).unsqueeze(1)

    infini.ops.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        None,
        None,
        None,
        None,
        cache_seqlens_tensor,
        None,
        None,
        block_table,
        alibi_slopes,
        0.125,
        True,
        (-1, -1),
        0.0,
        True,
        0,
        False,
        out,
        None,
        stream=torch.musa.current_stream().musa_stream,
        implementation_index=implementation_index,
    )

    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_cache, k_before, rtol=0, atol=0)
    torch.testing.assert_close(v_cache, v_before, rtol=0, atol=0)


@pytest.mark.parametrize("implementation_index", (8,))
def test_moore_paged_flash_attn_with_kvcache_uses_stream_and_current_lengths(
    device, implementation_index
):
    if device != "musa":
        pytest.skip("paged Moore stream coverage requires the Moore backend")

    head_dim = 64
    q = torch.randn((2, 1, 4, head_dim), dtype=torch.float16, device=device)
    k_cache = torch.randn((3, 256, 2, head_dim), dtype=torch.float16, device=device)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.tensor(((2, -1), (1, 0)), dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor((0, 257), dtype=torch.int32, device=device)
    out = torch.full_like(q, math.nan)
    expected = _reference_varlen_attention(
        q[1:, 0],
        k_cache,
        v_cache,
        (1,),
        (257,),
        None,
        True,
        (-1, -1),
        block_table[1:],
    ).cpu()
    updated_block_table = torch.tensor(
        ((0, -1), (2, -1)), dtype=torch.int32, device=device
    )
    updated_cache_seqlens = torch.tensor((1, 256), dtype=torch.int32, device=device)
    updated_expected = _reference_varlen_attention(
        q[:, 0],
        k_cache,
        v_cache,
        (1, 1),
        (1, 256),
        None,
        True,
        (-1, -1),
        updated_block_table,
    ).cpu()
    torch.musa.synchronize()

    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    torch.musa._sleep(50_000_000)
    try:
        infini.ops.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            None,
            None,
            None,
            None,
            cache_seqlens,
            None,
            None,
            block_table,
            None,
            None,
            True,
            (-1, -1),
            0.0,
            True,
            0,
            False,
            out,
            None,
            stream=stream.musa_stream,
            implementation_index=implementation_index,
        )

        stream.synchronize()
        with torch.musa.stream(stream):
            actual = out.cpu()
        torch.testing.assert_close(actual[1, 0], expected[0], rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            actual[0], torch.zeros_like(actual[0]), rtol=0, atol=0
        )

        with torch.musa.stream(stream):
            cache_seqlens.copy_(updated_cache_seqlens)
            block_table.copy_(updated_block_table)
            out.fill_(math.nan)
        infini.ops.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            None,
            None,
            None,
            None,
            cache_seqlens,
            None,
            None,
            block_table,
            None,
            None,
            True,
            (-1, -1),
            0.0,
            True,
            0,
            False,
            out,
            None,
            stream=stream.musa_stream,
            implementation_index=implementation_index,
        )

        stream.synchronize()
        with torch.musa.stream(stream):
            updated_actual = out.cpu()
        torch.testing.assert_close(
            updated_actual[:, 0], updated_expected, rtol=1e-2, atol=1e-2
        )
    finally:
        torch.musa.synchronize()
