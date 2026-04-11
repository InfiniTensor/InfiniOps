import infini.ops
import pytest
import torch

from tests.utils import Payload


def _reshape_and_cache_ref(key, value, kv_cache, slot_mapping, kv_cache_out):
    """Reference implementation: scatter key/value into paged KV cache."""
    kv_cache_out.copy_(kv_cache)
    num_tokens = key.size(0)

    for i in range(num_tokens):
        slot = slot_mapping[i].item()

        if slot < 0:
            continue

        block_size = kv_cache_out.size(2)
        block_idx = slot // block_size
        block_offset = slot % block_size

        # kv_cache_out shape: [2, num_blocks, block_size, num_kv_heads, head_size]
        kv_cache_out[0, block_idx, block_offset, :, :] = key[i]
        kv_cache_out[1, block_idx, block_offset, :, :] = value[i]

    return kv_cache_out


def _reshape_and_cache(key, value, kv_cache, slot_mapping, kv_cache_out):
    infini.ops.reshape_and_cache(key, value, kv_cache, slot_mapping, kv_cache_out)

    return kv_cache_out


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "num_tokens, num_kv_heads, head_size, num_blocks, block_size",
    (
        (1, 1, 64, 1, 1),
        (4, 8, 64, 4, 16),
        (7, 4, 128, 8, 32),
        (16, 32, 128, 16, 16),
        (3, 2, 64, 2, 8),
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
def test_reshape_and_cache(
    num_tokens, num_kv_heads, head_size, num_blocks, block_size, dtype, device,
    rtol, atol
):
    total_slots = num_blocks * block_size

    if num_tokens > total_slots:
        pytest.skip("more tokens than available slots")

    key = torch.randn(
        num_tokens, num_kv_heads, head_size, dtype=dtype, device=device
    )
    value = torch.randn(
        num_tokens, num_kv_heads, head_size, dtype=dtype, device=device
    )

    kv_cache = torch.zeros(
        2, num_blocks, block_size, num_kv_heads, head_size,
        dtype=dtype, device=device,
    )

    # Build a slot mapping: assign each token a unique random slot.
    slots = torch.randperm(total_slots)[:num_tokens].to(
        dtype=torch.int64, device=device
    )

    kv_cache_out = kv_cache.clone()

    return Payload(
        _reshape_and_cache,
        _reshape_and_cache_ref,
        (key, value, kv_cache, slots, kv_cache_out),
        {},
        rtol=rtol,
        atol=atol,
    )
