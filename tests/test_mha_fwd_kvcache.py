import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "batch_size, kv_len, num_heads, num_kv_heads, head_size",
    (
        (1, 30, 7, 1, 128),
        (2, 17, 8, 2, 64),
        (3, 23, 16, 4, 128),
    ),
)
@pytest.mark.parametrize("seqlens_on_cpu", (False, True))
@pytest.mark.parametrize("block_size", (64, 128, 256))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_mha_fwd_kvcache_paged_decode(
    batch_size,
    kv_len,
    num_heads,
    num_kv_heads,
    head_size,
    seqlens_on_cpu,
    block_size,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    blocks_per_seq = (kv_len + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_seq
    scale = 1.0 / head_size**0.5

    q = randn_strided(
        (batch_size, 1, num_heads, head_size), None, dtype=dtype, device=device
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
    block_table = torch.empty(
        (batch_size, blocks_per_seq), dtype=torch.int32, device=device
    )

    for batch in range(batch_size):
        for block in range(blocks_per_seq):
            block_table[batch, block] = batch * blocks_per_seq + block

    seqlens_device = "cpu" if seqlens_on_cpu else device
    seqlens_k = torch.full(
        (batch_size,), kv_len, dtype=torch.int32, device=seqlens_device
    )

    return Payload(
        lambda *args, **kwargs: _mha_fwd_kvcache(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _ref_mha_fwd_kvcache,
        (q, kcache, vcache, seqlens_k, block_table, out),
        {
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "block_size": block_size,
            "scale": scale,
        },
        rtol=rtol,
        atol=atol,
    )


def _mha_fwd_kvcache(
    q,
    kcache,
    vcache,
    seqlens_k,
    block_table,
    out,
    *,
    num_heads,
    num_kv_heads,
    block_size,
    scale,
    implementation_index=0,
):
    del num_heads, num_kv_heads, block_size

    infini.ops.mha_fwd_kvcache(
        q,
        kcache,
        vcache,
        None,
        None,
        seqlens_k,
        None,
        None,
        None,
        None,
        block_table,
        None,
        out,
        scale,
        True,
        -1,
        0,
        0.0,
        False,
        0,
        implementation_index=implementation_index,
        stream=get_stream(q.device),
    )

    return out


def _ref_mha_fwd_kvcache(
    q,
    kcache,
    vcache,
    seqlens_k,
    block_table,
    out,
    *,
    num_heads,
    num_kv_heads,
    block_size,
    scale,
):
    del out

    q_cpu = q.cpu().float()
    kcache_cpu = kcache.cpu().float()
    vcache_cpu = vcache.cpu().float()
    seqlens_cpu = seqlens_k.cpu()
    block_table_cpu = block_table.cpu()
    outputs = []

    for batch in range(q.shape[0]):
        kv_len = int(seqlens_cpu[batch].item())
        remaining = kv_len
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
            q_cpu[batch].transpose(0, 1).unsqueeze(0),
            k_i.unsqueeze(0),
            v_i.unsqueeze(0),
            scale=scale,
            is_causal=False,
        )
        outputs.append(ref.squeeze(0).transpose(0, 1).unsqueeze(0).to(q.dtype))

    return torch.cat(outputs, dim=0).to(q.device)
