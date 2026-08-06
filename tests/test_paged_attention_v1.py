import math

import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream


def _get_alibi_slopes(num_heads):
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = [base**i for i in range(1, closest_power_of_2 + 1)]
    if num_heads > closest_power_of_2:
        powers += [
            base ** (i * 2)
            for i in range(1, 2 * (num_heads - closest_power_of_2) + 1, 2)
        ]

    return powers[:num_heads]


def _reference(query, key_cache, value_cache, block_tables, seq_lens, alibi, scale):
    output = torch.empty_like(query)
    num_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    queries_per_kv = num_heads // num_kv_heads
    block_size = key_cache.shape[3]

    for seq_id in range(query.shape[0]):
        seq_len = seq_lens[seq_id].item()
        keys = []
        values = []
        for token_idx in range(seq_len):
            block_id = block_tables[seq_id, token_idx // block_size].item()
            block_offset = token_idx % block_size
            keys.append(
                key_cache[block_id, :, :, block_offset, :].reshape(num_kv_heads, -1)
            )
            values.append(value_cache[block_id, :, :, block_offset])

        key = torch.stack(keys)
        value = torch.stack(values)
        if queries_per_kv > 1:
            key = torch.repeat_interleave(key, queries_per_kv, dim=1)
            value = torch.repeat_interleave(value, queries_per_kv, dim=1)

        scores = torch.einsum("hd,khd->hk", query[seq_id], key).float() * scale
        if alibi is not None:
            positions = torch.arange(seq_len, device=query.device)
            scores += alibi.view(-1, 1) * (positions - seq_len + 1)

        weights = torch.softmax(scores, dim=-1).to(query.dtype)
        output[seq_id] = torch.einsum("hk,khd->hd", weights, value)

    return output


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "num_seqs",
        "num_heads",
        "num_kv_heads",
        "head_size",
        "block_size",
        "max_seq_len",
        "use_alibi",
    ),
    (
        (1, 1, 1, 64, 16, 128, False),
        (3, 8, 2, 128, 16, 256, False),
        (2, 4, 2, 64, 8, 64, True),
    ),
)
@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-2, 1e-3),
        (torch.bfloat16, 5e-2, 5e-3),
    ),
)
def test_paged_attention_v1(
    num_seqs,
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    max_seq_len,
    use_alibi,
    index_dtype,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    scale = head_size**-0.5
    max_blocks_per_seq = math.ceil(max_seq_len / block_size)
    num_blocks = num_seqs * max_blocks_per_seq
    key_cache_x = 16 // torch.empty((), dtype=dtype).element_size()

    query = torch.randn((num_seqs, num_heads, head_size), dtype=dtype, device=device)
    key_cache = torch.randn(
        (
            num_blocks,
            num_kv_heads,
            head_size // key_cache_x,
            block_size,
            key_cache_x,
        ),
        dtype=dtype,
        device=device,
    )
    value_cache = torch.randn(
        (num_blocks, num_kv_heads, head_size, block_size),
        dtype=dtype,
        device=device,
    )
    block_tables = torch.arange(num_blocks, dtype=index_dtype, device=device).view(
        num_seqs, max_blocks_per_seq
    )
    seq_lens = torch.randint(
        1, max_seq_len + 1, (num_seqs,), dtype=index_dtype, device=device
    )
    alibi = (
        torch.tensor(_get_alibi_slopes(num_heads), dtype=torch.float32, device=device)
        if use_alibi
        else None
    )
    out = torch.empty_like(query)

    args = (
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        alibi,
        num_kv_heads,
        scale,
        block_size,
        max_seq_len,
        "auto",
        1.0,
        1.0,
        0,
        0,
        0,
        64,
        0,
    )

    return Payload(
        lambda *call_args, **kwargs: _paged_attention_v1(
            *call_args,
            **kwargs,
            implementation_index=implementation_index,
        ),
        _torch_paged_attention_v1,
        args,
        {"out": out},
        rtol=rtol,
        atol=atol,
    )


def _paged_attention_v1(*args, out, implementation_index):
    infini.ops.paged_attention_v1(
        *args,
        out,
        implementation_index=implementation_index,
        stream=get_stream(args[0].device),
    )

    return out


def _torch_paged_attention_v1(
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    alibi,
    num_kv_heads,
    scale,
    block_size,
    max_seq_len,
    kv_cache_dtype,
    k_scale,
    v_scale,
    tp_rank,
    blocksparse_local_blocks,
    blocksparse_vert_stride,
    blocksparse_block_size,
    blocksparse_head_sliding_step,
    *,
    out,
):
    assert num_kv_heads == key_cache.shape[1]
    assert block_size == key_cache.shape[3]
    assert max_seq_len <= block_tables.shape[1] * block_size
    assert kv_cache_dtype == "auto"
    assert k_scale == 1.0 and v_scale == 1.0
    assert tp_rank == 0
    assert blocksparse_local_blocks == 0
    assert blocksparse_vert_stride == 0
    assert blocksparse_block_size == 64
    assert blocksparse_head_sliding_step == 0

    out.copy_(
        _reference(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            alibi,
            scale,
        )
    )

    return out
