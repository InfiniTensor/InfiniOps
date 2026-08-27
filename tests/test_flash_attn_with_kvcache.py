import math

import infini.ops
import pytest
import torch

from tests.utils import get_stream


try:
    import flash_attn
except ImportError:
    flash_attn = None


if not hasattr(infini.ops, "FlashAttnWithKvcache"):
    pytest.skip(
        "`FlashAttnWithKvcache` is not available on this platform",
        allow_module_level=True,
    )


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
    if device not in ("cuda", "mlu"):
        pytest.skip("FlashAttention FA2 requires the NVIDIA or Cambricon backend")
    if device == "cuda" and implementation_index == 0:
        pytest.skip("Iluvatar native provider supports paged decode only")
    if device == "cuda" and flash_attn is None:
        pytest.skip("flash_attn is required as the dense reference")

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
    if device == "mlu":
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
        expected, expected_softmax_lse = flash_attn.flash_attn_with_kvcache(
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
    if device not in ("cuda", "mlu"):
        pytest.skip("FlashAttention FA2 requires the NVIDIA or Cambricon backend")

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
    if device == "mlu" or (device == "cuda" and implementation_index == 0):
        expected, _ = _reference_flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            causal=True,
        )
    else:
        if flash_attn is None:
            pytest.skip("flash_attn is required as the paged reference")
        expected = flash_attn.flash_attn_with_kvcache(
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
    if device == "cuda" and flash_attn is None:
        pytest.skip("flash_attn is required as the dense reference")

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
        expected = flash_attn.flash_attn_with_kvcache(
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
    if device == "cuda" and flash_attn is None:
        pytest.skip("flash_attn is required as the dense reference")

    q = torch.randn((2, 1, 4, 64), dtype=torch.float16, device=device)
    k_cache = torch.randn((2, 8, 2, 64), dtype=torch.float16, device=device)
    v_cache = torch.randn_like(k_cache)
    if device == "mlu":
        expected, _ = _reference_flash_attn_with_kvcache(q, k_cache, v_cache)
    else:
        expected = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache)
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
    else:
        pytest.skip("stream coverage requires an accelerator backend")
    if device == "cuda" and implementation_index == 0:
        pytest.skip("Iluvatar native provider supports paged decode only")
    if device == "cuda" and flash_attn is None:
        pytest.skip("flash_attn is required as the dense reference")

    q = torch.randn((2, 1, 4, 64), dtype=torch.float16, device=device)
    k_cache = torch.randn((2, 8, 2, 64), dtype=torch.float16, device=device)
    v_cache = torch.randn_like(k_cache)
    if device == "mlu":
        expected, _ = _reference_flash_attn_with_kvcache(q, k_cache, v_cache)
    else:
        expected = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache)
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
