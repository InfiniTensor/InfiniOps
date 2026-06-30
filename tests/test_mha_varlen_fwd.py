import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "seq_lens, num_heads, num_kv_heads, head_size",
    (
        ((16,), 8, 8, 64),
        ((7, 11, 5), 8, 2, 64),
        ((9, 3), 16, 4, 128),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_mha_varlen_fwd(
    seq_lens,
    num_heads,
    num_kv_heads,
    head_size,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    scale = 1.0 / head_size**0.5
    total_tokens = sum(seq_lens)
    q = randn_strided(
        (total_tokens, num_heads, head_size), None, dtype=dtype, device=device
    )
    k = randn_strided(
        (total_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    v = randn_strided(
        (total_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    out = torch.empty_like(q)
    cu = torch.tensor(
        [0] + [sum(seq_lens[: i + 1]) for i in range(len(seq_lens))],
        dtype=torch.int32,
        device=device,
    )

    return Payload(
        lambda *args, **kwargs: _mha_varlen_fwd(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _ref_mha_varlen_fwd,
        (q, k, v, out, cu, cu),
        {
            "seq_lens": seq_lens,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "scale": scale,
            "causal": True,
        },
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "seq_lens, num_heads, num_kv_heads, head_size",
    (
        ((5, 3), 8, 2, 64),
        ((7,), 16, 4, 128),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_mha_varlen_fwd_paged_kv(
    seq_lens,
    num_heads,
    num_kv_heads,
    head_size,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    block_size = 256
    batch_size = len(seq_lens)
    blocks_per_seq = (max(seq_lens) + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_seq
    scale = 1.0 / head_size**0.5
    total_tokens = sum(seq_lens)

    q = randn_strided(
        (total_tokens, num_heads, head_size), None, dtype=dtype, device=device
    )
    kcache = randn_strided(
        (num_blocks, block_size, num_kv_heads, head_size),
        None,
        dtype=dtype,
        device=device,
    )
    vcache = randn_strided(
        (num_blocks, block_size, num_kv_heads, head_size),
        None,
        dtype=dtype,
        device=device,
    )
    out = torch.empty_like(q)
    cu = torch.tensor(
        [0] + [sum(seq_lens[: i + 1]) for i in range(batch_size)],
        dtype=torch.int32,
        device=device,
    )
    block_table = torch.empty(
        (batch_size, blocks_per_seq), dtype=torch.int32, device=device
    )

    for batch in range(batch_size):
        for block in range(blocks_per_seq):
            block_table[batch, block] = batch * blocks_per_seq + block

    return Payload(
        lambda *args, **kwargs: _mha_varlen_fwd(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _ref_mha_varlen_fwd_paged_kv,
        (q, kcache, vcache, out, cu, cu, block_table),
        {
            "seq_lens": seq_lens,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "block_size": block_size,
            "scale": scale,
            "causal": True,
        },
        rtol=rtol,
        atol=atol,
    )


def _mha_varlen_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    block_table=None,
    *,
    seq_lens,
    num_heads,
    num_kv_heads,
    head_size,
    block_size=None,
    scale,
    causal,
    implementation_index=0,
):
    del block_size

    infini.ops.mha_varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        block_table,
        None,
        max(seq_lens),
        max(seq_lens),
        0.0,
        scale,
        False,
        causal,
        -1,
        0,
        0.0,
        False,
        None,
        0,
        implementation_index=implementation_index,
        stream=get_stream(q.device),
    )

    return out


def _ref_mha_varlen_fwd_paged_kv(
    q,
    kcache,
    vcache,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    block_table,
    *,
    seq_lens,
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    scale,
    causal,
):
    del out, cu_seqlens_q, cu_seqlens_k, head_size

    q_cpu = q.cpu().float()
    kcache_cpu = kcache.cpu().float()
    vcache_cpu = vcache.cpu().float()
    block_table_cpu = block_table.cpu()
    outputs = []
    q_offset = 0

    for batch, seq_len in enumerate(seq_lens):
        remaining = seq_len
        k_pages = []
        v_pages = []

        for block in block_table_cpu[batch]:
            if remaining <= 0:
                break

            take = min(remaining, block_size)
            block_id = int(block.item())
            k_pages.append(kcache_cpu[block_id, :take].transpose(0, 1))
            v_pages.append(vcache_cpu[block_id, :take].transpose(0, 1))
            remaining -= take

        k_i = torch.cat(k_pages, dim=1)
        v_i = torch.cat(v_pages, dim=1)

        if num_kv_heads < num_heads:
            repeat = num_heads // num_kv_heads
            k_i = k_i.repeat_interleave(repeat, dim=0)
            v_i = v_i.repeat_interleave(repeat, dim=0)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q_cpu[q_offset : q_offset + seq_len].transpose(0, 1).unsqueeze(0),
            k_i.unsqueeze(0),
            v_i.unsqueeze(0),
            scale=scale,
            is_causal=causal,
        )
        outputs.append(ref.squeeze(0).transpose(0, 1).to(q.dtype))
        q_offset += seq_len

    return torch.cat(outputs, dim=0).to(q.device)


def _ref_mha_varlen_fwd(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    cu_seqlens_k,
    *,
    seq_lens,
    num_heads,
    num_kv_heads,
    head_size,
    scale,
    causal,
):
    del out, cu_seqlens_q, cu_seqlens_k, head_size

    outputs = []
    offset = 0

    for seq_len in seq_lens:
        q_i = q[offset : offset + seq_len].cpu().float().transpose(0, 1)
        k_i = k[offset : offset + seq_len].cpu().float().transpose(0, 1)
        v_i = v[offset : offset + seq_len].cpu().float().transpose(0, 1)

        if num_kv_heads < num_heads:
            repeat = num_heads // num_kv_heads
            k_i = k_i.repeat_interleave(repeat, dim=0)
            v_i = v_i.repeat_interleave(repeat, dim=0)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q_i.unsqueeze(0),
            k_i.unsqueeze(0),
            v_i.unsqueeze(0),
            scale=scale,
            is_causal=causal,
        )
        outputs.append(ref.squeeze(0).transpose(0, 1).to(q.dtype))
        offset += seq_len

    return torch.cat(outputs, dim=0).to(q.device)
