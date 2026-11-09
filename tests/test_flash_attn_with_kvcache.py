import ctypes
import math
import os
import shlex
from types import SimpleNamespace

import infini.ops
import pytest
import torch

from tests.utils import get_stream


if not hasattr(infini.ops, "FlashAttnWithKvcache"):
    pytest.skip(
        "`FlashAttnWithKvcache` is not available on this platform",
        allow_module_level=True,
    )


def _get_flash_attn():
    return pytest.importorskip("flash_attn")


def _reference_decode(q, k_cache, v_cache, cache_seqlens, block_table=None, scale=None):
    scale = q.size(-1) ** -0.5 if scale is None else scale
    group_size = q.size(2) // k_cache.size(2)
    outputs = []

    for batch, length in enumerate(cache_seqlens):
        if block_table is None:
            key = k_cache[batch, :length]
            value = v_cache[batch, :length]
        else:
            blocks = block_table[batch].cpu().long()
            key = k_cache[blocks].flatten(0, 1)[:length]
            value = v_cache[blocks].flatten(0, 1)[:length]

        key = key.repeat_interleave(group_size, dim=1)
        value = value.repeat_interleave(group_size, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[batch].float(), key.float())
        probs = torch.softmax(scores * scale, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probs, value.float()))

    return torch.stack(outputs).to(q.dtype)


@pytest.mark.parametrize("cache_seqlens_kind", ("tensor", "scalar"))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 2e-2, 2e-2),
    ),
)
def test_flash_attn_with_kvcache_ascend_dense(
    cache_seqlens_kind, dtype, device, implementation_index, rtol, atol
):
    if device != "npu":
        pytest.skip("this test covers the Ascend provider")

    batch_size, cache_size = 2, 16
    num_heads, num_kv_heads, head_size = 4, 2, 64
    q = torch.randn((batch_size, 1, num_heads, head_size), dtype=dtype, device=device)
    k_cache = torch.randn(
        (batch_size, cache_size, num_kv_heads, head_size),
        dtype=dtype,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    lengths = (5, 13) if cache_seqlens_kind == "tensor" else (5, 5)
    cache_seqlens = (
        torch.tensor(lengths, dtype=torch.int32, device=device)
        if cache_seqlens_kind == "tensor"
        else lengths[0]
    )
    expected = _reference_decode(q, k_cache, v_cache, lengths, scale=0.125)
    actual = torch.empty_like(q)

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
        None,
        None,
        0.125,
        True,
        (-1, -1),
        0.0,
        True,
        0,
        False,
        actual,
        None,
        stream=get_stream(q.device),
        implementation_index=implementation_index,
    )

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_flash_attn_with_kvcache_ascend_paged(device, implementation_index):
    if device != "npu":
        pytest.skip("this test covers the Ascend provider")

    batch_size, page_size = 2, 256
    num_heads, num_kv_heads, head_size = 4, 2, 64
    q = torch.randn(
        (batch_size, 1, num_heads, head_size),
        dtype=torch.float16,
        device=device,
    )
    k_cache = torch.randn(
        (4, page_size, num_kv_heads, head_size),
        dtype=torch.float16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    lengths = (130, 300)
    cache_seqlens = torch.tensor(lengths, dtype=torch.int32, device=device)
    block_table = torch.tensor(((0, 1), (2, 3)), dtype=torch.int32, device=device)
    expected = _reference_decode(q, k_cache, v_cache, lengths, block_table=block_table)
    actual = torch.empty_like(q)

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
        actual,
        None,
        stream=get_stream(q.device),
        implementation_index=implementation_index,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_flash_attn_with_kvcache_ascend_non_default_stream(
    device, implementation_index
):
    if device != "npu":
        pytest.skip("this test covers the Ascend provider")

    q = torch.randn((2, 1, 4, 64), dtype=torch.float16, device=device)
    k_cache = torch.randn((2, 8, 2, 64), dtype=torch.float16, device=device)
    v_cache = torch.randn_like(k_cache)
    expected = _reference_decode(q, k_cache, v_cache, (8, 8))
    actual = torch.empty_like(q)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())

    infini.ops.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        actual,
        stream=stream.npu_stream,
        implementation_index=implementation_index,
    )

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("cache_seqlens_kind", ("tensor", "scalar"))
@pytest.mark.parametrize("append_kv", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 2e-2, 2e-2),
    ),
)
def test_flash_attn_with_kvcache_dense(
    cache_seqlens_kind,
    append_kv,
    dtype,
    device,
    implementation_index,
    rtol,
    atol,
):
    if device not in ("cuda", "musa", "mlu"):
        pytest.skip(
            "FlashAttention FA2 requires the NVIDIA, Moore, or Cambricon backend"
        )
    if device == "cuda" and implementation_index == 0:
        pytest.skip("Iluvatar native provider supports paged decode only")
    if device == "musa" and implementation_index != 16:
        pytest.skip("dense attention requires the Mate linked provider")

    batch_size, cache_size = 2, 16
    num_heads, num_kv_heads, head_size = 4, 2, 64
    q = torch.randn((batch_size, 2, num_heads, head_size), dtype=dtype, device=device)
    k_cache = torch.randn(
        (batch_size, cache_size, num_kv_heads, head_size),
        dtype=dtype,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    k = (
        torch.randn(
            (batch_size, 2, num_kv_heads, head_size),
            dtype=dtype,
            device=device,
        )
        if append_kv
        else None
    )
    v = torch.randn_like(k) if k is not None else None
    cache_seqlens = (
        torch.full((batch_size,), 5, dtype=torch.int32, device=device)
        if cache_seqlens_kind == "tensor"
        else 5
    )
    expected_k_cache = k_cache.clone()
    expected_v_cache = v_cache.clone()
    actual_k_cache = k_cache.clone()
    actual_v_cache = v_cache.clone()
    if device in ("musa", "mlu"):
        expected, expected_softmax_lse = _reference_flash_attn_with_kvcache(
            q,
            expected_k_cache,
            expected_v_cache,
            k,
            v,
            cache_seqlens=cache_seqlens,
            softmax_scale=0.125,
            causal=True,
            window_size=(4, 0),
        )
    else:
        expected, expected_softmax_lse = _get_flash_attn().flash_attn_with_kvcache(
            q,
            expected_k_cache,
            expected_v_cache,
            k,
            v,
            cache_seqlens=cache_seqlens,
            softmax_scale=0.125,
            causal=True,
            window_size=(4, 0),
            num_splits=1,
            return_softmax_lse=True,
        )
    actual = torch.empty_like(q)
    actual_softmax_lse = torch.empty(
        (q.size(0), q.size(2), q.size(1)),
        dtype=torch.float32,
        device=q.device,
    )

    infini.ops.flash_attn_with_kvcache(
        q,
        actual_k_cache,
        actual_v_cache,
        k,
        v,
        None,
        None,
        cache_seqlens,
        None,
        None,
        None,
        None,
        0.125,
        True,
        (4, 0),
        0.0,
        True,
        1,
        True,
        actual,
        actual_softmax_lse,
        stream=get_stream(q.device),
        implementation_index=implementation_index,
    )

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        actual_softmax_lse,
        expected_softmax_lse,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(actual_k_cache, expected_k_cache, rtol=0, atol=0)
    torch.testing.assert_close(actual_v_cache, expected_v_cache, rtol=0, atol=0)


@pytest.mark.smoke
def test_flash_attn_with_kvcache_paged(device, implementation_index):
    if device not in ("cuda", "musa", "mlu"):
        pytest.skip(
            "FlashAttention FA2 requires the NVIDIA, Moore, or Cambricon backend"
        )

    batch_size, page_size = 2, 256
    num_heads, num_kv_heads, head_size = 4, 2, 64
    q = torch.randn(
        (batch_size, 1, num_heads, head_size),
        dtype=torch.float16,
        device=device,
    )
    k_cache = torch.randn(
        (4, page_size, num_kv_heads, head_size),
        dtype=torch.float16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    cache_seqlens = torch.tensor((130, 300), dtype=torch.int32, device=device)
    block_table = torch.tensor(((0, 1), (2, 3)), dtype=torch.int32, device=device)
    if device in ("musa", "mlu") or (device == "cuda" and implementation_index == 0):
        expected, _ = _reference_flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=True,
        )
    else:
        expected = _get_flash_attn().flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=True,
        )
    actual = torch.empty_like(q)

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
        actual,
        None,
        stream=get_stream(q.device),
        implementation_index=implementation_index,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_flash_attn_with_kvcache_scalar_seqlens_with_cache_batch_idx(
    device, implementation_index
):
    if device not in ("cuda", "mlu"):
        pytest.skip("FlashAttention FA2 requires the NVIDIA or Cambricon backend")
    if device == "cuda" and implementation_index == 0:
        pytest.skip("Iluvatar native provider requires tensor cache lengths")

    q = torch.randn((2, 1, 4, 60), dtype=torch.float16, device=device)
    k_cache = torch.randn((3, 8, 2, 60), dtype=torch.float16, device=device)
    v_cache = torch.randn_like(k_cache)
    cache_batch_idx = torch.tensor((2, 0), dtype=torch.int32, device=device)
    # Dao's scalar wrapper sizes by cache batch, so use the query-batch tensor
    # expected by its C++ kernel when `cache_batch_idx` remaps a smaller batch.
    if device == "mlu":
        expected, _ = _reference_flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=5,
            cache_batch_idx=cache_batch_idx,
        )
    else:
        expected = _get_flash_attn().flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=torch.full((q.size(0),), 5, dtype=torch.int32, device=device),
            cache_batch_idx=cache_batch_idx,
        )
    actual = torch.empty_like(q)

    infini.ops.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        None,
        None,
        None,
        None,
        5,
        cache_batch_idx,
        None,
        None,
        None,
        None,
        False,
        (-1, -1),
        0.0,
        True,
        0,
        False,
        actual,
        None,
        stream=get_stream(q.device),
        implementation_index=implementation_index,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_flash_attn_with_kvcache_defaults(device, implementation_index):
    if device not in ("cuda", "mlu"):
        pytest.skip("FlashAttention FA2 requires the NVIDIA or Cambricon backend")
    if device == "cuda" and implementation_index == 0:
        pytest.skip("Iluvatar native provider supports paged decode only")

    q = torch.randn((2, 1, 4, 64), dtype=torch.float16, device=device)
    k_cache = torch.randn((2, 8, 2, 64), dtype=torch.float16, device=device)
    v_cache = torch.randn_like(k_cache)
    if device == "mlu":
        expected, _ = _reference_flash_attn_with_kvcache(q, k_cache, v_cache)
    else:
        expected = _get_flash_attn().flash_attn_with_kvcache(q, k_cache, v_cache)
    actual = torch.empty_like(q)

    infini.ops.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        actual,
        stream=get_stream(q.device),
        implementation_index=implementation_index,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_flash_attn_with_kvcache_non_default_stream(device, implementation_index):
    if device == "cuda":
        accelerator = torch.cuda
        stream_attribute = "cuda_stream"
    elif device == "mlu":
        accelerator = torch.mlu
        stream_attribute = "mlu_stream"
    elif device == "musa" and implementation_index == 16:
        accelerator = torch.musa
        stream_attribute = "musa_stream"
    else:
        pytest.skip("stream coverage requires an accelerator backend")
    if device == "cuda" and implementation_index == 0:
        pytest.skip("Iluvatar native provider supports paged decode only")

    q = torch.randn((2, 1, 4, 64), dtype=torch.float16, device=device)
    k_cache = torch.randn((2, 8, 2, 64), dtype=torch.float16, device=device)
    v_cache = torch.randn_like(k_cache)
    if device in ("musa", "mlu"):
        expected, _ = _reference_flash_attn_with_kvcache(q, k_cache, v_cache)
    else:
        expected = _get_flash_attn().flash_attn_with_kvcache(q, k_cache, v_cache)
    actual = torch.empty_like(q)
    stream = accelerator.Stream()
    stream.wait_stream(accelerator.current_stream())

    infini.ops.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        actual,
        stream=getattr(stream, stream_attribute),
        implementation_index=implementation_index,
    )

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


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
def test_flash_attn_with_kvcache_paged_moore_decode_matrix(
    device, implementation_index, num_kv_heads, head_dim, dtype, rtol, atol
):
    if device != "musa":
        pytest.skip("paged decode matrix requires the Moore backend")

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

    expected, _ = _reference_flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens_tensor,
        block_table=block_table,
        softmax_scale=0.125,
        causal=True,
        alibi_slopes=alibi_slopes,
    )

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
        stream=get_stream(q.device),
        implementation_index=implementation_index,
    )

    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_cache, k_before, rtol=0, atol=0)
    torch.testing.assert_close(v_cache, v_before, rtol=0, atol=0)


@pytest.mark.parametrize("implementation_index", (8,))
def test_flash_attn_with_kvcache_paged_moore_uses_stream_and_current_metadata(
    device, implementation_index
):
    if device != "musa":
        pytest.skip("paged stream metadata coverage requires the Moore backend")

    accelerator = torch.musa
    head_dim = 64
    q = torch.randn((2, 1, 4, head_dim), dtype=torch.float16, device=device)
    k_cache = torch.randn((3, 256, 2, head_dim), dtype=torch.float16, device=device)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.tensor(((2, -1), (1, 0)), dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor((0, 257), dtype=torch.int32, device=device)
    out = torch.full_like(q, math.nan)
    expected, _ = _reference_flash_attn_with_kvcache(
        q[1:],
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens[1:],
        block_table=block_table[1:],
        causal=True,
    )
    expected = expected.cpu()
    updated_block_table = torch.tensor(
        ((0, -1), (2, -1)), dtype=torch.int32, device=device
    )
    updated_cache_seqlens = torch.tensor((1, 256), dtype=torch.int32, device=device)
    updated_expected, _ = _reference_flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=updated_cache_seqlens,
        block_table=updated_block_table,
        causal=True,
    )
    updated_expected = updated_expected.cpu()
    accelerator.synchronize()

    stream = accelerator.Stream()
    stream.wait_stream(accelerator.current_stream())

    # Keep the current stream busy after the target dependency is recorded. A
    # provider that ignores the raw stream then leaves output untouched at target sync.
    accelerator._sleep(50_000_000)
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
        with accelerator.stream(stream):
            actual = out.cpu()
        torch.testing.assert_close(actual[1], expected[0], rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            actual[0], torch.zeros_like(actual[0]), rtol=0, atol=0
        )

        with accelerator.stream(stream):
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
        with accelerator.stream(stream):
            updated_actual = out.cpu()
        torch.testing.assert_close(
            updated_actual, updated_expected, rtol=1e-2, atol=1e-2
        )
    finally:
        accelerator.synchronize()


def _reference_flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
):
    batch_size, query_length, num_heads, _ = q.shape
    if cache_seqlens is None:
        lengths = [k_cache.size(1)] * batch_size
    elif isinstance(cache_seqlens, int):
        lengths = [cache_seqlens] * batch_size
    else:
        lengths = cache_seqlens.cpu().tolist()
    cache_rows = (
        list(range(batch_size))
        if cache_batch_idx is None
        else cache_batch_idx.cpu().tolist()
    )
    append_length = 0 if k is None else k.size(1)

    outputs = []
    softmax_lses = []
    for batch in range(batch_size):
        length = lengths[batch]
        if k is not None:
            row = cache_rows[batch]
            k_cache[row, length : length + append_length].copy_(k[batch])
            v_cache[row, length : length + append_length].copy_(v[batch])
        length += append_length

        if block_table is None:
            row = cache_rows[batch]
            k_seq = k_cache[row, :length]
            v_seq = v_cache[row, :length]
        else:
            page_size = k_cache.size(1)
            block_count = (length + page_size - 1) // page_size
            blocks = block_table[batch, :block_count].cpu().tolist()
            k_seq = torch.cat(tuple(k_cache[index] for index in blocks))[:length]
            v_seq = torch.cat(tuple(v_cache[index] for index in blocks))[:length]

        q_seq = q[batch].transpose(0, 1)
        k_seq = k_seq.transpose(0, 1)
        v_seq = v_seq.transpose(0, 1)
        groups = num_heads // k_seq.size(0)
        k_seq = k_seq.repeat_interleave(groups, dim=0)
        v_seq = v_seq.repeat_interleave(groups, dim=0)
        scale = softmax_scale if softmax_scale is not None else q.size(-1) ** -0.5
        scores = torch.matmul(q_seq.float(), k_seq.float().transpose(-2, -1))
        scores *= scale
        if alibi_slopes is not None:
            slopes = alibi_slopes if alibi_slopes.ndim == 1 else alibi_slopes[batch]
            query_positions = torch.arange(query_length, device=q.device).unsqueeze(1)
            key_positions = torch.arange(length, device=q.device).unsqueeze(0)
            distance = (query_positions + length - query_length - key_positions).abs()
            scores += -slopes[:, None, None] * distance
        mask = _attention_mask(
            query_length,
            length,
            causal,
            window_size,
            q.device,
        )
        if mask is not None:
            scores.masked_fill_(~mask.unsqueeze(0), -math.inf)
        softmax_lse = torch.logsumexp(scores, dim=-1)
        softmax_lses.append(
            torch.where(torch.isneginf(softmax_lse), math.inf, softmax_lse)
        )
        probabilities = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        outputs.append(torch.matmul(probabilities, v_seq.float()).to(q.dtype))

    return (
        torch.stack(outputs).transpose(1, 2),
        torch.stack(softmax_lses),
    )


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


_ILUVATAR_DECODE_CASES = (
    (1, 32, 2, 128, 4096, 1, 7, "head_first"),
    (2, 32, 2, 128, 4096, 1, 7, "head_first"),
    (1, 32, 2, 128, 4096, 1, 2049, "head_first"),
    (2, 32, 2, 128, 4096, 1, 2049, "head_first"),
    (1, 32, 2, 128, 4096, 1, 2910, "head_first"),
    (2, 32, 2, 128, 4096, 1, 2910, "head_first"),
    (1, 32, 2, 128, 4096, 1, 4096, "head_first"),
    (2, 32, 2, 128, 4096, 1, 4096, "head_first"),
    (1, 8, 1, 128, 256, 4, 777, "token_first"),
    (2, 16, 4, 128, 512, 4, 1537, "token_first"),
    (4, 32, 8, 128, 256, 16, 4096, "token_first"),
    (8, 16, 2, 128, 512, 8, 2910, "head_first"),
    (16, 8, 1, 128, 256, 8, 1025, "token_first"),
    (1, 64, 8, 128, 1024, 4, 4095, "head_first"),
    (2, 128, 16, 128, 2048, 2, 2049, "token_first"),
    (4, 8, 8, 128, 256, 4, 513, "token_first"),
    (1, 32, 2, 128, 256, 32, 8191, "token_first"),
    (4, 16, 4, 128, 256, 32, 8192, "head_first"),
    (4, 8, 2, 128, 512, 4, 2047, "padded_rows"),
    (2, 16, 2, 128, 256, 1, 1, "token_first"),
    (1, 8, 1, 128, 256, 1, 0, "token_first"),
)


def _iluvatar_decode_inputs(case, device, dtype=torch.bfloat16):
    batch, heads, kv_heads, dim, page, blocks, length, layout = case
    q = torch.randn((batch, 1, heads, dim), device=device, dtype=dtype)
    physical_blocks = batch * blocks + 2
    if layout == "head_first":
        k = torch.randn(
            (physical_blocks, kv_heads, page, dim), device=device, dtype=dtype
        ).permute(0, 2, 1, 3)
        v = torch.randn(
            (physical_blocks, kv_heads, page, dim), device=device, dtype=dtype
        ).permute(0, 2, 1, 3)
    else:
        k = torch.randn(
            (physical_blocks, page, kv_heads, dim), device=device, dtype=dtype
        )
        v = torch.randn_like(k)

    out = torch.empty_like(q)
    if layout == "padded_rows":
        padded_q = torch.empty((batch, 1, heads + 1, dim), device=device, dtype=dtype)
        padded_q[:, :, :heads].copy_(q)
        q = padded_q[:, :, :heads]
        out = torch.empty((batch, 1, heads + 2, dim), device=device, dtype=dtype)[
            :, :, :heads
        ]

    table = torch.arange(
        physical_blocks - 1, 1, -1, device=device, dtype=torch.int32
    ).reshape(batch, blocks)
    lengths = tuple(max(0, length - index) for index in range(batch))
    cache_seqlens = torch.tensor(lengths, dtype=torch.int32, device=device)
    return q, k, v, table, cache_seqlens, out, lengths


def _iluvatar_decode(q, k, v, table, lengths, out):
    # Explicit tensor metadata prevents a singleton from selecting the scalar overload.
    lengths_view = SimpleNamespace(
        data_ptr=lengths.data_ptr,
        shape=lengths.shape,
        dtype=lengths.dtype,
        device=lengths.device,
        stride=lengths.stride,
    )
    infini.ops.flash_attn_with_kvcache(
        q,
        k,
        v,
        None,
        None,
        None,
        None,
        lengths_view,
        None,
        None,
        table,
        None,
        q.size(-1) ** -0.5,
        True,
        (-1, -1),
        0.0,
        True,
        0,
        False,
        out,
        None,
        stream=get_stream(q.device),
        implementation_index=0,
    )


@pytest.mark.smoke
@pytest.mark.parametrize("case", _ILUVATAR_DECODE_CASES)
@pytest.mark.parametrize("implementation_index", (0,))
def test_flash_attn_with_kvcache_iluvatar_decode_reduction(
    case, device, implementation_index
):
    if (
        device != "cuda"
        or not infini.ops.FlashAttnWithKvcache.active_implementation_indices("iluvatar")
    ):
        pytest.skip("requires the Iluvatar backend")

    q, k, v, table, lengths, out, host_lengths = _iluvatar_decode_inputs(case, device)
    expected = _reference_decode(q, k, v, host_lengths, block_table=table)
    _iluvatar_decode(q, k, v, table, lengths, out)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-3)
    assert torch.isfinite(out).all()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "case,dtype",
    (
        ((2, 8, 2, 64, 256, 4, 777, "token_first"), torch.bfloat16),
        ((2, 8, 2, 128, 512, 4, 1537, "head_first"), torch.float16),
        ((17, 8, 2, 128, 256, 4, 777, "token_first"), torch.bfloat16),
    ),
)
@pytest.mark.parametrize("implementation_index", (0,))
def test_flash_attn_with_kvcache_iluvatar_decode_fallback(
    case, dtype, device, implementation_index
):
    if (
        device != "cuda"
        or not infini.ops.FlashAttnWithKvcache.active_implementation_indices("iluvatar")
    ):
        pytest.skip("requires the Iluvatar backend")

    q, k, v, table, lengths, out, host_lengths = _iluvatar_decode_inputs(
        case, device, dtype
    )
    expected = _reference_decode(q, k, v, host_lengths, block_table=table)
    _iluvatar_decode(q, k, v, table, lengths, out)
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-3)


class _DecodeTensorView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.c_int64 * 4),
        ("strides", ctypes.c_int64 * 4),
        ("rank", ctypes.c_int),
        ("dtype", ctypes.c_int),
    ]


def _build_iluvatar_decode_bridge(directory):
    from tests.test_cpp_api import _compiler, _install_prefixes, _library_dir, _run

    prefixes = _install_prefixes()
    library_dirs = list(
        dict.fromkeys(
            _library_dir(prefixes, name)
            for name in ("libinfiniops.so", "libinfinirt.so")
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "flash_decode.cc"
    library = directory / "flash_decode.so"
    source.write_text(_ILUVATAR_WORKSPACE_SOURCE)
    _run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-shared",
            "-fPIC",
            *shlex.split(os.environ.get("CXXFLAGS", "")),
            *(f"-I{prefix / 'include'}" for prefix in prefixes),
            str(source),
            *(f"-L{path}" for path in library_dirs),
            "-linfiniops",
            "-linfinirt",
            *(f"-Wl,-rpath,{path}" for path in library_dirs),
            "-o",
            str(library),
        ]
    )
    bridge = ctypes.CDLL(str(library))
    bridge.flash_decode.argtypes = [
        ctypes.POINTER(_DecodeTensorView),
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    bridge.flash_decode.restype = None

    return bridge


@pytest.fixture(scope="module")
def _iluvatar_decode_bridge(tmp_path_factory):
    if not infini.ops.FlashAttnWithKvcache.active_implementation_indices("iluvatar"):
        pytest.skip("requires the Iluvatar backend")

    return _build_iluvatar_decode_bridge(tmp_path_factory.mktemp("flash_decode"))


def _iluvatar_workspace_call(bridge, q, k, v, table, lengths, out):
    tensors = (q, k, v, table, lengths, out)
    dtype_ids = {torch.bfloat16: 0, torch.float16: 1, torch.int32: 2}
    views = (_DecodeTensorView * len(tensors))(
        *(
            _DecodeTensorView(
                tensor.data_ptr(),
                (ctypes.c_int64 * 4)(*tensor.shape),
                (ctypes.c_int64 * 4)(*tensor.stride()),
                tensor.ndim,
                dtype_ids[tensor.dtype],
            )
            for tensor in tensors
        )
    )
    workspace = torch.empty(
        8 * q.size(0) * q.size(2) * (q.size(3) + 2),
        dtype=torch.float32,
        device=q.device,
    )

    def call():
        bridge.flash_decode(
            views,
            q.device.index,
            workspace.data_ptr(),
            workspace.numel() * workspace.element_size(),
            get_stream(q.device),
        )

    return call


@pytest.mark.parametrize("case", _ILUVATAR_DECODE_CASES)
@pytest.mark.parametrize("implementation_index", (0,))
def test_flash_attn_with_kvcache_iluvatar_workspace_decode(
    case, device, implementation_index, _iluvatar_decode_bridge
):
    if device != "cuda":
        pytest.skip("requires the Iluvatar backend")

    q, k, v, table, lengths, out, host_lengths = _iluvatar_decode_inputs(case, device)
    expected = _reference_decode(q, k, v, host_lengths, block_table=table)
    call = _iluvatar_workspace_call(
        _iluvatar_decode_bridge, q, k, v, table, lengths, out
    )
    call()
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-3)
    assert torch.isfinite(out).all()


_ILUVATAR_WORKSPACE_SOURCE = r"""
#include <infini/ops.h>
#include <cmath>
#include <cstdint>

using namespace infini::ops;

struct DecodeTensorView {
  void* data;
  int64_t shape[4];
  int64_t strides[4];
  int rank;
  int dtype;
};

extern "C" void flash_decode(const DecodeTensorView* views, int device_index,
                             void* workspace, std::size_t workspace_bytes,
                             void* stream) {
  const Device device{Device::Type::kIluvatar, device_index};
  const auto tensor = [&](int index) {
    const auto& view = views[index];
    const auto dtype = view.dtype == 0 ? DataType::kBFloat16
                     : view.dtype == 1 ? DataType::kFloat16 : DataType::kInt32;
    return Tensor{view.data, Tensor::Shape(view.shape, view.shape + view.rank),
                  dtype, device,
                  Tensor::Strides(view.strides, view.strides + view.rank)};
  };
  const auto q = tensor(0), k = tensor(1), v = tensor(2);
  const std::optional<Tensor> table{tensor(3)}, lengths{tensor(4)};
  const auto out = tensor(5);
  const std::optional<Tensor> none;
  const std::optional<double> scale{1.0 / std::sqrt(double(q.size(3)))};
  Handle handle;
  handle.set_stream(stream);
  handle.set_workspace(workspace);
  handle.set_workspace_size_in_bytes(workspace_bytes);
  Config config;
  config.set_implementation_index(0);
  FlashAttnWithKvcache::Call(handle, config, q, k, v, none, none, none, none,
                           lengths, none, none, table, none, scale, true,
                           std::vector<int64_t>{-1, -1}, 0.0, true, int64_t{0},
                           false, out, none);
}
"""
