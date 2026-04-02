import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_npu_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_size",
    (
        (32, 32, 128),   # MHA
        (32, 8, 128),    # GQA (4x)
        (16, 4, 64),     # GQA (4x), smaller
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
@pytest.mark.parametrize("device", ("npu",))
def test_flash_attention_prefill_single(
    num_heads, num_kv_heads, head_size, dtype, rtol, atol, device,
):
    """Single sequence prefill (no block table)."""
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    num_tokens = 16
    scale = 1.0 / head_size**0.5

    query = randn_strided(
        (num_tokens, num_heads, head_size), None, dtype=dtype, device=device
    )
    key = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    value = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    output = torch.empty(
        (num_tokens, num_heads, head_size), dtype=dtype, device=device
    )

    return Payload(
        lambda q, k, v, o: _flash_attention(
            q, k, v, None, None, None,
            num_heads, num_kv_heads, head_size,
            scale, 3, 0, o,
        ),
        lambda q, k, v, o: _ref_flash_attention(
            q, k, v,
            num_heads, num_kv_heads, head_size,
            scale, causal=True,
        ),
        (query, key, value, output),
        {},
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_size",
    (
        (32, 8, 128),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
@pytest.mark.parametrize("device", ("npu",))
def test_flash_attention_prefill_multi(
    num_heads, num_kv_heads, head_size, dtype, rtol, atol, device,
):
    """Multi-sequence prefill with cu_seqlens."""
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    seq_lens = [8, 12, 4]
    num_tokens = sum(seq_lens)
    scale = 1.0 / head_size**0.5

    query = randn_strided(
        (num_tokens, num_heads, head_size), None, dtype=dtype, device=device
    )
    key = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    value = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    output = torch.empty(
        (num_tokens, num_heads, head_size), dtype=dtype, device=device
    )

    cu_seqlens_q = torch.tensor(
        [0] + [sum(seq_lens[:i+1]) for i in range(len(seq_lens))],
        dtype=torch.int64, device=device,
    )
    cu_seqlens_kv = cu_seqlens_q.clone()

    return Payload(
        lambda q, k, v, o: _flash_attention(
            q, k, v, None, cu_seqlens_q, cu_seqlens_kv,
            num_heads, num_kv_heads, head_size,
            scale, 3, 0, o,
        ),
        lambda q, k, v, o: _ref_flash_attention_multi(
            q, k, v,
            seq_lens, seq_lens,
            num_heads, num_kv_heads, head_size,
            scale, causal=True,
        ),
        (query, key, value, output),
        {},
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_size, block_size",
    (
        (32, 8, 128, 128),
        (16, 4, 64, 128),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
@pytest.mark.parametrize("device", ("npu",))
def test_flash_attention_decode(
    num_heads, num_kv_heads, head_size, block_size,
    dtype, rtol, atol, device,
):
    """Decode phase: single token per request with paged KV cache."""
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    num_reqs = 3
    kv_len = 16  # Total KV length per request.
    num_blocks_per_req = (kv_len + block_size - 1) // block_size
    num_blocks = num_reqs * num_blocks_per_req
    scale = 1.0 / head_size**0.5

    query = randn_strided(
        (num_reqs, num_heads, head_size), None, dtype=dtype, device=device
    )
    # Paged KV cache: [num_blocks, KV_N, block_size, D]
    kv_cache = randn_strided(
        (num_blocks, num_kv_heads, block_size, head_size),
        None, dtype=dtype, device=device,
    )
    output = torch.empty(
        (num_reqs, num_heads, head_size), dtype=dtype, device=device
    )

    # Block table: request i uses blocks [i*num_blocks_per_req, ...].
    block_table = torch.zeros(
        (num_reqs, num_blocks_per_req), dtype=torch.int32, device=device
    )
    for i in range(num_reqs):
        for j in range(num_blocks_per_req):
            block_table[i, j] = i * num_blocks_per_req + j

    cu_seqlens_q = torch.arange(
        0, num_reqs + 1, dtype=torch.int64, device=device
    )
    cu_seqlens_kv = torch.tensor(
        [0] + [kv_len] * num_reqs, dtype=torch.int64, device=device
    )

    return Payload(
        lambda q, k, v, o: _flash_attention(
            q, k, v, block_table, cu_seqlens_q, cu_seqlens_kv,
            num_heads, num_kv_heads, head_size,
            scale, 3, block_size, o,
        ),
        lambda q, k, v, o: _ref_flash_attention_paged(
            q, k, block_table,
            cu_seqlens_q, cu_seqlens_kv,
            num_heads, num_kv_heads, head_size,
            block_size, scale, causal=True,
        ),
        (query, kv_cache, kv_cache, output),
        {},
        rtol=rtol,
        atol=atol,
    )


def _flash_attention(
    query, key, value,
    block_table, cu_seqlens_q, cu_seqlens_kv,
    num_heads, num_kv_heads, head_size,
    scale, sparse_mode, block_size, output,
):
    if query.device.type == "npu":
        infini.ops.flash_attention(
            query, key, value,
            block_table, cu_seqlens_q, cu_seqlens_kv,
            num_heads, num_kv_heads, head_size,
            scale, sparse_mode, block_size, output, stream=get_npu_stream(query),
        )
    else:
        infini.ops.flash_attention(
            query, key, value,
            block_table, cu_seqlens_q, cu_seqlens_kv,
            num_heads, num_kv_heads, head_size,
            scale, sparse_mode, block_size, output,
        )

    return output


def _ref_flash_attention(query, key, value,
                         num_heads, num_kv_heads, head_size,
                         scale, causal=True):
    """PyTorch SDPA reference for single-sequence prefill."""
    # [T, N, D] -> [N, T, D]
    q = query.transpose(0, 1).float()
    k = key.transpose(0, 1).float()
    v = value.transpose(0, 1).float()

    # GQA: expand K/V to match num_heads.
    if num_kv_heads < num_heads:
        ratio = num_heads // num_kv_heads
        k = k.repeat_interleave(ratio, dim=0)
        v = v.repeat_interleave(ratio, dim=0)

    # [N, T, D] -> [1, N, T, D] for scaled_dot_product_attention.
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=causal
    )

    # [1, N, T, D] -> [T, N, D] -> original dtype.
    return out.squeeze(0).transpose(0, 1).to(query.dtype)


def _ref_flash_attention_multi(query, key, value,
                               seq_lens_q, seq_lens_kv,
                               num_heads, num_kv_heads, head_size,
                               scale, causal=True):
    """PyTorch SDPA reference for multi-sequence prefill."""
    outputs = []
    offset = 0
    for sq, sk in zip(seq_lens_q, seq_lens_kv):
        q = query[offset:offset + sq]
        k = key[offset:offset + sq]
        v = value[offset:offset + sq]
        out = _ref_flash_attention(
            q, k, v, num_heads, num_kv_heads, head_size, scale, causal
        )
        outputs.append(out)
        offset += sq

    return torch.cat(outputs, dim=0)


def _ref_flash_attention_paged(query, kv_cache_arg, block_table,
                               cu_seqlens_q, cu_seqlens_kv,
                               num_heads, num_kv_heads, head_size,
                               block_size, scale, causal=True):
    """PyTorch SDPA reference for decode with paged KV cache."""
    cu_q = cu_seqlens_q.cpu()
    cu_kv = cu_seqlens_kv.cpu()
    bt = block_table.cpu()
    cache = kv_cache_arg.cpu()
    num_reqs = bt.size(0)
    outputs = []

    for i in range(num_reqs):
        q = query[i:i+1]  # [1, N, D]
        kv_len = int(cu_kv[i + 1] - cu_kv[i])

        # Gather KV from paged cache.
        # cache: [num_blocks, KV_N, block_size, D]
        blocks = bt[i]
        k_pages = []
        v_pages = []
        remaining = kv_len
        for b in blocks:
            if remaining <= 0:
                break
            take = min(remaining, block_size)
            k_pages.append(cache[int(b.item()), :, :take, :])
            v_pages.append(cache[int(b.item()), :, :take, :])
            remaining -= take
        k = torch.cat(k_pages, dim=1)  # [KV_N, kv_len, D]
        v = torch.cat(v_pages, dim=1)

        out = _ref_flash_attention(
            q.transpose(0, 1),  # [1, N, D] -> [N, 1, D]
            k.transpose(0, 1),  # [KV_N, kv_len, D] -> [kv_len, KV_N, D]
            v.transpose(0, 1),
            num_heads, num_kv_heads, head_size, scale, causal
        )
        outputs.append(out)

    return torch.cat(outputs, dim=0)
