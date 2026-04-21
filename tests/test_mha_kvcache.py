import infini.ops
import pytest
import torch

from tests.utils import Payload, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "batch_size, seq_len_q, num_heads_q, num_heads_k, head_size, max_seq_k, block_size",
    (
        (1, 1, 8, 2, 32, 16, 4),
        (2, 1, 16, 2, 64, 32, 8),
        (4, 1, 32, 4, 64, 128, 16),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 5e-3, 5e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_mha_kvcache(
    batch_size,
    seq_len_q,
    num_heads_q,
    num_heads_k,
    head_size,
    max_seq_k,
    block_size,
    dtype,
    device,
    rtol,
    atol,
):
    active_indices = infini.ops.MhaKvcache.active_implementation_indices(device)

    if 1 not in active_indices:
        pytest.skip(f"implementation `1` not active on `{device}`")

    num_blocks_per_seq = (max_seq_k + block_size - 1) // block_size
    num_blocks = batch_size * num_blocks_per_seq + 1

    q = randn_strided((batch_size, seq_len_q, num_heads_q, head_size), None, dtype=dtype, device=device)
    k_cache = randn_strided((num_blocks, block_size, num_heads_k, head_size), None, dtype=dtype, device=device)
    v_cache = randn_strided((num_blocks, block_size, num_heads_k, head_size), None, dtype=dtype, device=device)

    # Per-sequence KV lengths: vary across the batch to exercise masking.
    seqlens_k = torch.tensor(
        [max_seq_k - i for i in range(batch_size)], dtype=torch.int32, device=device
    )

    # Assign contiguous block ids to each sequence.
    block_ids = torch.arange(1, 1 + batch_size * num_blocks_per_seq, dtype=torch.int32, device=device)
    block_table = block_ids.view(batch_size, num_blocks_per_seq)

    scale = 1.0 / (head_size**0.5)
    out = torch.empty((batch_size, seq_len_q, num_heads_q, head_size), dtype=dtype, device=device)

    return Payload(
        _mha_kvcache,
        _torch_mha_kvcache,
        (q, k_cache, v_cache, seqlens_k, block_table, scale, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _mha_kvcache(q, k_cache, v_cache, seqlens_k, block_table, scale, out):
    infini.ops.mha_kvcache(
        q, k_cache, v_cache, seqlens_k, block_table, scale, out,
        implementation_index=1,
    )

    return out


def _torch_mha_kvcache(q, k_cache, v_cache, seqlens_k, block_table, scale, out):
    batch_size, seq_len_q, num_heads_q, head_size = q.shape
    _, block_size, num_heads_k, _ = k_cache.shape
    max_len = int(seqlens_k.max().item())

    # Gather the first `max_len` K/V positions per sequence from the paged cache.
    pos = torch.arange(max_len, device=q.device, dtype=torch.long)
    block_idx = (pos // block_size).unsqueeze(0).expand(batch_size, -1)
    within = (pos % block_size).unsqueeze(0).expand(batch_size, -1)
    clamped_block_idx = block_idx.clamp(max=block_table.size(1) - 1)
    phys_blocks = block_table.long().gather(1, clamped_block_idx)
    k = k_cache[phys_blocks, within]
    v = v_cache[phys_blocks, within]

    # Build key-padding mask.
    key_mask = torch.arange(max_len, device=q.device).unsqueeze(0) < seqlens_k.long().unsqueeze(1)
    attn_mask = torch.zeros((batch_size, 1, 1, max_len), dtype=q.dtype, device=q.device)
    attn_mask = attn_mask.masked_fill(~key_mask.unsqueeze(1).unsqueeze(1), float("-inf"))

    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)

    result = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
        scale=scale, enable_gqa=True,
    )

    out.copy_(result.transpose(1, 2))

    return out
