import infini.ops
import pytest
import torch

from tests.utils import Payload, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "num_tokens, num_kv_heads, head_size, num_blocks, block_size, num_padding",
    (
        (4, 2, 8, 16, 4, 0),
        (17, 4, 16, 8, 8, 3),
        (64, 8, 64, 32, 16, 0),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 0, 0),
        (torch.float16, 0, 0),
        (torch.bfloat16, 0, 0),
    ),
)
def test_paged_caching(
    num_tokens,
    num_kv_heads,
    head_size,
    num_blocks,
    block_size,
    num_padding,
    dtype,
    device,
    rtol,
    atol,
):
    active_indices = infini.ops.PagedCaching.active_implementation_indices(device)

    if 1 not in active_indices:
        pytest.skip(f"implementation `1` not active on `{device}`")

    k = randn_strided((num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device)
    v = randn_strided((num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device)
    k_cache = torch.zeros((num_blocks, num_kv_heads, block_size, head_size), dtype=dtype, device=device)
    v_cache = torch.zeros((num_blocks, num_kv_heads, block_size, head_size), dtype=dtype, device=device)

    # Build a random slot assignment, optionally with padding (negative slots).
    total_slots = num_blocks * block_size
    assert num_tokens - num_padding <= total_slots
    slot_values = torch.randperm(total_slots)[: num_tokens - num_padding]
    padding = -torch.ones(num_padding, dtype=torch.int64)
    slot_mapping = torch.cat((slot_values.to(torch.int64), padding)).to(device)

    return Payload(
        _paged_caching,
        _torch_paged_caching,
        (k_cache, v_cache, k, v, slot_mapping),
        {},
        rtol=rtol,
        atol=atol,
    )


def _paged_caching(k_cache, v_cache, k, v, slot_mapping):
    infini.ops.paged_caching(k_cache, v_cache, k, v, slot_mapping, implementation_index=1)

    # Return a flattened concatenation so `auto_act_and_assert` can compare.
    return torch.cat((k_cache.reshape(-1), v_cache.reshape(-1)))


def _torch_paged_caching(k_cache, v_cache, k, v, slot_mapping):
    block_size = k_cache.shape[-2]
    valid = slot_mapping >= 0
    valid_slots = slot_mapping[valid]
    block_idx = valid_slots // block_size
    block_offset = valid_slots % block_size
    k_cache[block_idx, :, block_offset, :] = k[valid]
    v_cache[block_idx, :, block_offset, :] = v[valid]

    return torch.cat((k_cache.reshape(-1), v_cache.reshape(-1)))
