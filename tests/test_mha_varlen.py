import infini.ops
import pytest
import torch

from tests.utils import Payload, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "seqlens, num_heads_q, num_heads_k, head_size, block_size",
    (
        ((8,), 8, 2, 32, 4),
        ((5, 7), 16, 2, 64, 8),
        ((11, 9, 13), 32, 4, 64, 16),
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
def test_mha_varlen(
    seqlens,
    num_heads_q,
    num_heads_k,
    head_size,
    block_size,
    dtype,
    device,
    rtol,
    atol,
):
    active_indices = infini.ops.MhaVarlen.active_implementation_indices(device)

    if 1 not in active_indices:
        pytest.skip(f"implementation `1` not active on `{device}`")

    batch_size = len(seqlens)
    total_q = sum(seqlens)

    # Allocate enough blocks so the `batch_size` sequences fit; add slack.
    blocks_per_seq = max((s + block_size - 1) // block_size for s in seqlens)
    num_blocks = batch_size * blocks_per_seq + 1

    q = randn_strided((total_q, num_heads_q, head_size), None, dtype=dtype, device=device)
    k_cache = randn_strided((num_blocks, block_size, num_heads_k, head_size), None, dtype=dtype, device=device)
    v_cache = randn_strided((num_blocks, block_size, num_heads_k, head_size), None, dtype=dtype, device=device)

    cum = [0]
    for s in seqlens:
        cum.append(cum[-1] + s)

    cum_seqlens_q = torch.tensor(cum, dtype=torch.int32, device=device)
    cum_seqlens_k = torch.tensor(cum, dtype=torch.int32, device=device)

    # Give each sequence a contiguous block range starting at block `1`.
    block_ids = torch.arange(1, 1 + batch_size * blocks_per_seq, dtype=torch.int32, device=device)
    block_table = block_ids.view(batch_size, blocks_per_seq)

    scale = 1.0 / (head_size**0.5)
    out = torch.empty_like(q)

    return Payload(
        _mha_varlen,
        _torch_mha_varlen,
        (q, k_cache, v_cache, cum_seqlens_q, cum_seqlens_k, block_table, scale, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _mha_varlen(q, k_cache, v_cache, cum_seqlens_q, cum_seqlens_k, block_table, scale, out):
    infini.ops.mha_varlen(
        q, k_cache, v_cache, cum_seqlens_q, cum_seqlens_k, block_table,
        scale, out,
        implementation_index=1,
    )

    return out


def _torch_mha_varlen(q, k_cache, v_cache, cum_seqlens_q, cum_seqlens_k, block_table, scale, out):
    block_size = k_cache.shape[1]
    cu_q = cum_seqlens_q.to("cpu").to(torch.int64).tolist()
    cu_k = cum_seqlens_k.to("cpu").to(torch.int64).tolist()
    batch_size = len(cu_q) - 1

    for b in range(batch_size):
        q_start, q_end = cu_q[b], cu_q[b + 1]
        k_start, k_end = cu_k[b], cu_k[b + 1]
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

        if seqlen_q == 0 or seqlen_k == 0:
            continue

        q_b = q[q_start:q_end]
        pos = torch.arange(seqlen_k, device=q.device, dtype=torch.long)
        block_idx = (pos // block_size).clamp(max=block_table.size(1) - 1)
        within = pos % block_size
        phys_blocks = block_table[b].long().gather(0, block_idx)
        k_b = k_cache[phys_blocks, within]
        v_b = v_cache[phys_blocks, within]

        q_sdpa = q_b.transpose(0, 1).unsqueeze(0)
        k_sdpa = k_b.transpose(0, 1).unsqueeze(0)
        v_sdpa = v_b.transpose(0, 1).unsqueeze(0)

        if seqlen_k == seqlen_q:
            result = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, dropout_p=0.0, is_causal=True,
                scale=scale, enable_gqa=True,
            )
        else:
            rows = torch.arange(seqlen_q, device=q.device, dtype=torch.long)
            cols = torch.arange(seqlen_k, device=q.device, dtype=torch.long)
            allowed = cols.unsqueeze(0) <= (rows.unsqueeze(1) + (seqlen_k - seqlen_q))
            attn_mask = torch.zeros((seqlen_q, seqlen_k), dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))
            attn_mask = attn_mask.view(1, 1, seqlen_q, seqlen_k)
            result = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, dropout_p=0.0,
                is_causal=False, scale=scale, enable_gqa=True,
            )

        out[q_start:q_end] = result.squeeze(0).transpose(0, 1)

    return out
