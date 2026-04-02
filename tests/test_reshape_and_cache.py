import infini.ops
import pytest
import torch

from tests.utils import Payload, get_npu_stream, randn_strided

# ReshapeAndCache only works on NPU (aclrtMemcpy-based), so tests only
# parametrize on float16/bfloat16 and use explicit device parametrization.


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "num_tokens, num_kv_heads, head_size, num_blocks, block_size",
    (
        (1, 8, 128, 4, 16),
        (4, 8, 128, 4, 16),
        (8, 4, 64, 8, 32),
        (16, 2, 128, 8, 64),
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
def test_reshape_and_cache_contiguous(
    num_tokens, num_kv_heads, head_size, num_blocks, block_size,
    dtype, rtol, atol, device,
):
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    key = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    value = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    kv_cache = torch.zeros(
        (num_blocks, block_size, num_kv_heads, head_size),
        dtype=dtype, device=device,
    )
    # Contiguous slot mapping: token i -> slot i.
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    return Payload(
        _reshape_and_cache,
        _ref_reshape_and_cache,
        (key, value, kv_cache, slot_mapping, kv_cache),
        {},
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "num_tokens, num_kv_heads, head_size, num_blocks, block_size",
    (
        (4, 8, 128, 4, 16),
        (8, 4, 64, 8, 32),
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
def test_reshape_and_cache_noncontiguous_slots(
    num_tokens, num_kv_heads, head_size, num_blocks, block_size,
    dtype, rtol, atol, device,
):
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    key = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    value = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    kv_cache = torch.zeros(
        (num_blocks, block_size, num_kv_heads, head_size),
        dtype=dtype, device=device,
    )
    # Non-contiguous slots: skip every other slot.
    slot_mapping = torch.tensor(
        [i * 2 for i in range(num_tokens)], dtype=torch.int64, device=device
    )

    return Payload(
        _reshape_and_cache,
        _ref_reshape_and_cache,
        (key, value, kv_cache, slot_mapping, kv_cache),
        {},
        rtol=rtol,
        atol=atol,
    )


def _reshape_and_cache(key, value, kv_cache, slot_mapping, kv_cache_out):
    if key.device.type == "npu":
        infini.ops.reshape_and_cache(key, value, kv_cache, slot_mapping, kv_cache_out, stream=get_npu_stream(key))
    else:
        infini.ops.reshape_and_cache(key, value, kv_cache, slot_mapping, kv_cache_out)

    return kv_cache_out


def _ref_reshape_and_cache(key, value, kv_cache, slot_mapping, kv_cache_out):
    kv_cache_out = kv_cache_out.clone()
    slots = slot_mapping.cpu()
    block_size = kv_cache_out.size(1)
    for i in range(key.size(0)):
        slot = int(slots[i].item())
        if slot < 0:
            continue
        block_idx = slot // block_size
        offset = slot % block_size
        kv_cache_out[block_idx, offset, :, :] = key[i, :, :]
    return kv_cache_out
