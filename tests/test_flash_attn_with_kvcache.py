import infini.ops
import pytest
import torch

from tests.utils import get_stream


flash_attn = pytest.importorskip("flash_attn")


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
    if device != "cuda":
        pytest.skip("FlashAttention FA2 requires the NVIDIA backend")

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


def test_flash_attn_with_kvcache_paged(device, implementation_index):
    if device != "cuda":
        pytest.skip("FlashAttention FA2 requires the NVIDIA backend")

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
    if device != "cuda":
        pytest.skip("FlashAttention FA2 requires the NVIDIA backend")

    q = torch.randn((2, 1, 4, 60), dtype=torch.float16, device=device)
    k_cache = torch.randn((3, 8, 2, 60), dtype=torch.float16, device=device)
    v_cache = torch.randn_like(k_cache)
    cache_batch_idx = torch.tensor((2, 0), dtype=torch.int32, device=device)
    # Dao's scalar wrapper sizes by cache batch, so use the query-batch tensor
    # expected by its C++ kernel when `cache_batch_idx` remaps a smaller batch.
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
    if device != "cuda":
        pytest.skip("FlashAttention FA2 requires the NVIDIA backend")

    q = torch.randn((2, 1, 4, 64), dtype=torch.float16, device=device)
    k_cache = torch.randn((2, 8, 2, 64), dtype=torch.float16, device=device)
    v_cache = torch.randn_like(k_cache)
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
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    q = torch.randn((2, 1, 4, 64), dtype=torch.float16, device=device)
    k_cache = torch.randn((2, 8, 2, 64), dtype=torch.float16, device=device)
    v_cache = torch.randn_like(k_cache)
    expected = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache)
    actual = torch.empty_like(q)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    infini.ops.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        actual,
        stream=stream.cuda_stream,
        implementation_index=implementation_index,
    )

    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
