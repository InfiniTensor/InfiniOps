import math

import infini.ops
import pytest
import torch

from tests.utils import get_stream


if not hasattr(infini.ops, "FlashAttnVarlenFunc"):
    pytest.skip(
        "`FlashAttnVarlenFunc` is not available on this platform",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    (
        "q_lens, k_lens, num_heads, num_kv_heads, causal, window_size, "
        "scale, paged, use_alibi"
    ),
    (
        ((3, 5), (4, 5), 4, 4, False, (-1, -1), None, False, False),
        ((5, 2), (3, 6), 4, 2, True, (-1, -1), 0.125, False, False),
        ((4, 3), (6, 2), 4, 2, False, (2, 1), None, False, False),
        ((4, 3), (6, 2), 4, 2, True, (2, 1), None, False, False),
        ((2, 3), (130, 300), 4, 2, True, (-1, -1), None, True, True),
    ),
)
@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 2e-2, 2e-2),
    ),
)
def test_flash_attn_varlen_func(
    q_lens,
    k_lens,
    num_heads,
    num_kv_heads,
    causal,
    window_size,
    scale,
    paged,
    use_alibi,
    head_dim,
    dtype,
    device,
    implementation_index,
    rtol,
    atol,
):
    if device != "cuda":
        pytest.skip("FlashAttention FA2 requires the NVIDIA backend")

    if (paged or use_alibi) and implementation_index == 8:
        pytest.skip("paged KV cache and ALiBi require the linked provider")

    q = torch.randn((sum(q_lens), num_heads, head_dim), dtype=dtype, device=device)
    block_table = None
    if paged:
        page_size = 256
        max_blocks = max((length + page_size - 1) // page_size for length in k_lens)
        block_rows = []
        num_blocks = 0

        for length in k_lens:
            blocks = (length + page_size - 1) // page_size
            row = list(range(num_blocks, num_blocks + blocks))
            row.extend([0] * (max_blocks - blocks))
            block_rows.append(row)
            num_blocks += blocks

        k = torch.randn(
            (num_blocks, page_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        block_table = torch.tensor(
            block_rows,
            dtype=torch.int32,
            device=device,
        )
    else:
        k = torch.randn(
            (sum(k_lens), num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

    v = torch.randn_like(k)
    cu_seqlens_q = _cumulative_lengths(q_lens, device)
    cu_seqlens_k = _cumulative_lengths(k_lens, device)
    alibi_slopes = (
        torch.linspace(0.01, 0.04, num_heads, dtype=torch.float32, device=device)
        if use_alibi
        else None
    )
    out = torch.empty_like(q)
    return_attn_probs = not paged
    softmax_lse = (
        torch.empty(
            (q.size(1), q.size(0)),
            dtype=torch.float32,
            device=q.device,
        )
        if return_attn_probs
        else None
    )
    s_dmask = (
        torch.empty((0,), dtype=q.dtype, device=q.device) if return_attn_probs else None
    )

    infini.ops.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        alibi_slopes,
        block_table,
        max(q_lens),
        max(k_lens),
        0.0,
        scale,
        causal,
        window_size,
        0.0,
        False,
        return_attn_probs,
        out,
        softmax_lse,
        s_dmask,
        stream=get_stream(q.device),
        implementation_index=implementation_index,
    )

    expected = _reference_varlen_attention(
        q,
        k,
        v,
        q_lens,
        k_lens,
        scale,
        causal,
        window_size,
        block_table,
        alibi_slopes,
    )
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)

    if return_attn_probs:
        expected_auxiliary = torch.ops.aten._flash_attention_forward.default(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max(q_lens),
            max(k_lens),
            0.0,
            causal,
            False,
            scale=scale,
            window_size_left=None if window_size[0] < 0 else window_size[0],
            window_size_right=(
                0 if causal else None if window_size[1] < 0 else window_size[1]
            ),
        )
        expected_softmax_lse = _pack_varlen_softmax_lse(
            expected_auxiliary[1],
            q_lens,
        )
        torch.testing.assert_close(softmax_lse, expected_softmax_lse)
        torch.testing.assert_close(s_dmask, expected_auxiliary[4])


def test_flash_attn_varlen_func_non_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    dtype = torch.float16
    q_lens = (3, 5)
    k_lens = (4, 5)
    q = torch.randn((sum(q_lens), 4, 64), dtype=dtype, device=device)
    k = torch.randn((sum(k_lens), 2, 64), dtype=dtype, device=device)
    v = torch.randn_like(k)
    cu_seqlens_q = _cumulative_lengths(q_lens, device)
    cu_seqlens_k = _cumulative_lengths(k_lens, device)
    out = torch.empty_like(q)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    infini.ops.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        max(q_lens),
        max(k_lens),
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
        stream=stream.cuda_stream,
        implementation_index=implementation_index,
    )

    stream.synchronize()
    expected = _reference_varlen_attention(
        q,
        k,
        v,
        q_lens,
        k_lens,
        None,
        True,
        (-1, -1),
    )
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)


def test_flash_attn_varlen_func_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("CUDA stream coverage requires the NVIDIA backend")

    q = torch.randn((5, 4, 64), dtype=torch.float16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu_seqlens = _cumulative_lengths((2, 3), device)
    out = torch.full_like(q, math.nan)
    current_stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    with torch.cuda.stream(current_stream):
        torch.cuda._sleep(100_000_000)
        infini.ops.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            3,
            3,
            out,
            implementation_index=implementation_index,
        )

    torch.cuda.default_stream().synchronize()
    snapshot = out.clone()
    current_stream.synchronize()
    expected = _reference_varlen_attention(
        q,
        k,
        v,
        (2, 3),
        (2, 3),
        None,
        False,
        (-1, -1),
    )
    torch.testing.assert_close(snapshot, expected, rtol=2e-3, atol=2e-3)


def test_flash_attn_varlen_func_defaults(device, implementation_index):
    if device != "cuda":
        pytest.skip("FlashAttention FA2 requires the NVIDIA backend")

    q = torch.randn((5, 4, 64), dtype=torch.float16, device=device)
    k = torch.randn((5, 4, 64), dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    cu_seqlens = _cumulative_lengths((2, 3), device)
    out = torch.empty_like(q)

    infini.ops.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        3,
        3,
        out,
        stream=get_stream(q.device),
        implementation_index=implementation_index,
    )

    expected = _reference_varlen_attention(
        q,
        k,
        v,
        (2, 3),
        (2, 3),
        None,
        False,
        (-1, -1),
    )
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)


def test_flash_attn_varlen_func_device_guard():
    if torch.cuda.device_count() < 2:
        pytest.skip("device-guard coverage requires at least two NVIDIA GPUs")

    original_device = torch.cuda.current_device()
    device = torch.device("cuda:1" if original_device == 0 else "cuda:0")
    q = torch.randn((5, 4, 64), dtype=torch.float16, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu_seqlens = _cumulative_lengths((2, 3), device)
    out = torch.empty_like(q)

    infini.ops.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        3,
        3,
        out,
        stream=get_stream(device),
    )

    torch.cuda.synchronize(device)
    assert torch.cuda.current_device() == original_device
    expected = _reference_varlen_attention(
        q,
        k,
        v,
        (2, 3),
        (2, 3),
        None,
        False,
        (-1, -1),
    )
    torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)


def _cumulative_lengths(lengths, device):
    values = [0]

    for length in lengths:
        values.append(values[-1] + length)

    return torch.tensor(values, dtype=torch.int32, device=device)


def _pack_varlen_softmax_lse(softmax_lse, q_lens):
    if softmax_lse.ndim == 2:
        return softmax_lse

    return torch.cat(
        tuple(
            sequence_lse[:, :q_len] for sequence_lse, q_len in zip(softmax_lse, q_lens)
        ),
        dim=1,
    )


def _reference_varlen_attention(
    q,
    k,
    v,
    q_lens,
    k_lens,
    scale,
    causal,
    window_size,
    block_table=None,
    alibi_slopes=None,
):
    outputs = []
    q_offset = 0
    k_offset = 0

    for batch_index, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
        q_seq = q[q_offset : q_offset + q_len].transpose(0, 1)
        if block_table is None:
            k_seq = k[k_offset : k_offset + k_len]
            v_seq = v[k_offset : k_offset + k_len]
        else:
            blocks = (k_len + k.size(1) - 1) // k.size(1)
            block_indices = block_table[batch_index, :blocks].tolist()
            k_seq = torch.cat(tuple(k[index] for index in block_indices))[:k_len]
            v_seq = torch.cat(tuple(v[index] for index in block_indices))[:k_len]

        k_seq = k_seq.transpose(0, 1)
        v_seq = v_seq.transpose(0, 1)
        groups = q_seq.size(0) // k_seq.size(0)
        k_seq = k_seq.repeat_interleave(groups, dim=0)
        v_seq = v_seq.repeat_interleave(groups, dim=0)
        mask = _attention_mask(
            q_len,
            k_len,
            causal,
            window_size,
            q.device,
        )
        scale_factor = scale if scale is not None else 1.0 / math.sqrt(q.size(-1))
        scores = (
            torch.matmul(q_seq.float(), k_seq.float().transpose(-2, -1)) * scale_factor
        )
        if alibi_slopes is not None:
            slopes = (
                alibi_slopes if alibi_slopes.ndim == 1 else alibi_slopes[batch_index]
            )
            query_positions = torch.arange(q_len, device=q.device).unsqueeze(1)
            key_positions = torch.arange(k_len, device=q.device).unsqueeze(0)
            distance = (query_positions + k_len - q_len - key_positions).abs()
            scores += -slopes[:, None, None] * distance

        if mask is not None:
            scores.masked_fill_(~mask.unsqueeze(0), -math.inf)
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = torch.nan_to_num(probabilities, nan=0.0)
        output = torch.matmul(probabilities, v_seq.float()).to(q.dtype)
        outputs.append(output.transpose(0, 1))
        q_offset += q_len
        k_offset += k_len

    return torch.cat(outputs)


def _attention_mask(q_len, k_len, causal, window_size, device):
    left, right = window_size

    if not causal and left < 0 and right < 0:
        return None

    query_positions = torch.arange(q_len, device=device).unsqueeze(1)
    key_positions = torch.arange(k_len, device=device).unsqueeze(0)
    aligned_query_positions = query_positions + k_len - q_len
    mask = torch.ones((q_len, k_len), dtype=torch.bool, device=device)

    if left >= 0:
        mask &= key_positions >= aligned_query_positions - left

    if causal:
        right = 0

    if right >= 0:
        mask &= key_positions <= aligned_query_positions + right

    return mask
