import math

import infini.ops
import pytest
import torch

from tests.test_flash_attn_varlen_func import (
    _cumulative_lengths,
    _reference_varlen_attention,
)


if not hasattr(infini.ops, "FlashAttnVarlenFunc"):
    pytest.skip(
        "`FlashAttnVarlenFunc` is not available on this platform",
        allow_module_level=True,
    )


@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("implementation_index", (8,))
def test_moore_paged_flash_attn_uses_stream_and_current_lengths(
    device, implementation_index, head_dim
):
    if device != "musa":
        pytest.skip("paged Moore stream coverage requires the Moore backend")

    q_lens = (2, 3)
    initial_k_lens = (130, 300)
    updated_k_lens = (65, 129)
    num_heads = 4
    num_kv_heads = 2
    page_size = 256

    # Sliced padded storage keeps the head dimension contiguous while making
    # every outer stride non-default. K and V deliberately use different
    # padding so their stride metadata cannot be interchanged.
    q = torch.randn(
        (sum(q_lens), num_heads, head_dim + 5),
        dtype=torch.float16,
        device=device,
    )[..., :head_dim]
    k = torch.randn(
        (4, page_size, num_kv_heads, head_dim + 7),
        dtype=torch.float16,
        device=device,
    )[..., :head_dim]
    v = torch.randn(
        (4, page_size, num_kv_heads, head_dim + 11),
        dtype=torch.float16,
        device=device,
    )[..., :head_dim]
    # Physical block 3 is intentionally unused and the first row's tail is
    # invalid. This catches providers that ignore the block table or scan
    # unused entries.
    block_table = torch.tensor(((2, -1), (1, 0)), dtype=torch.int32, device=device)
    cu_seqlens_q = _cumulative_lengths(q_lens, device)
    cu_seqlens_k = _cumulative_lengths(initial_k_lens, device)
    updated_cu_seqlens_k = _cumulative_lengths(updated_k_lens, device)
    out = torch.full(
        (sum(q_lens), num_heads, head_dim + 13),
        math.nan,
        dtype=torch.float16,
        device=device,
    )[..., :head_dim]

    expected = _reference_varlen_attention(
        q,
        k,
        v,
        q_lens,
        initial_k_lens,
        None,
        True,
        (-1, -1),
        block_table,
    ).cpu()
    torch.musa.synchronize()

    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())

    # Keep the current stream busy. A provider that ignores the explicit raw
    # stream will leave `out` untouched when the target stream is synchronized.
    torch.musa._sleep(50_000_000)
    try:
        infini.ops.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            block_table,
            max(q_lens),
            max(initial_k_lens),
            0.0,
            None,
            True,
            (-1, -1),
            0.0,
            False,
            False,
            out,
            None,
            None,
            stream=stream.musa_stream,
            implementation_index=implementation_index,
        )

        stream.synchronize()
        with torch.musa.stream(stream):
            actual = out.cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

        torch.musa.synchronize()
        expected = _reference_varlen_attention(
            q,
            k,
            v,
            q_lens,
            updated_k_lens,
            None,
            True,
            (-1, -1),
            block_table,
        ).cpu()

        with torch.musa.stream(stream):
            cu_seqlens_k.copy_(updated_cu_seqlens_k)
            out.fill_(math.nan)

        # Omit implementation_index on the replay: the Moore default must be
        # the same paged-capable mixed provider used above.
        infini.ops.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            block_table,
            max(q_lens),
            max(initial_k_lens),
            0.0,
            None,
            True,
            (-1, -1),
            0.0,
            False,
            False,
            out,
            None,
            None,
            stream=stream.musa_stream,
        )

        stream.synchronize()
        with torch.musa.stream(stream):
            actual = out.cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    finally:
        torch.musa.synchronize()


@pytest.mark.parametrize("implementation_index", (8,))
def test_moore_paged_flash_attn_empty_q(device, implementation_index):
    if device != "musa":
        pytest.skip("empty paged Moore coverage requires the Moore backend")

    q = torch.empty((0, 4, 64), dtype=torch.float16, device=device)
    k = torch.empty((1, 256, 2, 64), dtype=torch.float16, device=device)
    v = torch.empty_like(k)
    cu_seqlens_q = torch.tensor((0, 0), dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor((0, 0), dtype=torch.int32, device=device)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    out = torch.empty_like(q)

    infini.ops.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        block_table,
        1,
        1,
        0.0,
        None,
        True,
        (-1, -1),
        0.0,
        False,
        False,
        out,
        None,
        None,
        stream=torch.musa.current_stream().musa_stream,
        implementation_index=implementation_index,
    )

    assert out.numel() == 0
