import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream, randn_strided

# ReshapeAndCache only works on NPU (aclrtMemcpy-based), so tests only
# parametrize on float16/bfloat16 and use explicit device parametrization.

# `aclnnScatterPaKvCache` (index 1) requires Atlas A5 (SoC 260).  It compiles
# on 910B (CANN 8.5.1 headers present) but produces wrong results at runtime.
_SKIP_INDEX_1 = pytest.mark.skip(
    reason="`aclnnScatterPaKvCache` (index 1) requires Atlas A5; "
    "not supported on Ascend 910B"
)


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
    "implementation_index",
    (0, pytest.param(1, marks=_SKIP_INDEX_1), 2),
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
    num_tokens,
    num_kv_heads,
    head_size,
    num_blocks,
    block_size,
    implementation_index,
    dtype,
    rtol,
    atol,
    device,
):
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    active_indices = infini.ops.ReshapeAndCache.active_implementation_indices(device)

    if implementation_index not in active_indices:
        pytest.skip(f"implementation `{implementation_index}` not active on `{device}`")

    key = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    value = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    # Layout: [2, num_blocks, block_size, num_kv_heads, head_size]
    # Index 0 = key cache, index 1 = value cache.
    kv_cache = torch.zeros(
        (2, num_blocks, block_size, num_kv_heads, head_size),
        dtype=dtype,
        device=device,
    )
    # Contiguous slot mapping: token i -> slot i.
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    return Payload(
        lambda *args, **kwargs: _reshape_and_cache(
            *args, **kwargs, implementation_index=implementation_index
        ),
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
    "implementation_index",
    (0, pytest.param(1, marks=_SKIP_INDEX_1), 2),
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
    num_tokens,
    num_kv_heads,
    head_size,
    num_blocks,
    block_size,
    implementation_index,
    dtype,
    rtol,
    atol,
    device,
):
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    active_indices = infini.ops.ReshapeAndCache.active_implementation_indices(device)

    if implementation_index not in active_indices:
        pytest.skip(f"implementation `{implementation_index}` not active on `{device}`")

    key = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    value = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    kv_cache = torch.zeros(
        (2, num_blocks, block_size, num_kv_heads, head_size),
        dtype=dtype,
        device=device,
    )
    # Non-contiguous slots: skip every other slot.
    slot_mapping = torch.tensor(
        [i * 2 for i in range(num_tokens)], dtype=torch.int64, device=device
    )

    return Payload(
        lambda *args, **kwargs: _reshape_and_cache(
            *args, **kwargs, implementation_index=implementation_index
        ),
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
        (8, 8, 128, 4, 16),
        (16, 4, 64, 8, 32),
    ),
)
@pytest.mark.parametrize(
    "implementation_index",
    (0, pytest.param(1, marks=_SKIP_INDEX_1), 2),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
@pytest.mark.parametrize("device", ("npu",))
def test_reshape_and_cache_padding_slots(
    num_tokens,
    num_kv_heads,
    head_size,
    num_blocks,
    block_size,
    implementation_index,
    dtype,
    rtol,
    atol,
    device,
):
    """Graph-padded decode: slots with `-1` must be skipped, not written.

    `aclnnInplaceIndexCopy` silently treats `slot=-1` as "last index" which
    corrupts the last KV cache entry.  The wrapper must filter `-1` slots
    before calling the underlying op.
    """
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    active_indices = infini.ops.ReshapeAndCache.active_implementation_indices(device)

    if implementation_index not in active_indices:
        pytest.skip(f"implementation `{implementation_index}` not active on `{device}`")

    key = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    value = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    kv_cache = torch.zeros(
        (2, num_blocks, block_size, num_kv_heads, head_size),
        dtype=dtype,
        device=device,
    )

    # Every other token is a padding slot (`-1`); valid slots map to unique
    # contiguous positions so a correct wrapper leaves the final entry of
    # the last block untouched.
    slot_values = []
    valid = 0

    for i in range(num_tokens):
        if i % 2 == 0:
            slot_values.append(-1)
        else:
            slot_values.append(valid)
            valid += 1

    slot_mapping = torch.tensor(slot_values, dtype=torch.int64, device=device)

    return Payload(
        lambda *args, **kwargs: _reshape_and_cache(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _ref_reshape_and_cache,
        (key, value, kv_cache, slot_mapping, kv_cache),
        {},
        rtol=rtol,
        atol=atol,
    )


def _reshape_and_cache(
    key, value, kv_cache, slot_mapping, kv_cache_out, implementation_index=0
):
    infini.ops.reshape_and_cache(
        key,
        value,
        kv_cache,
        slot_mapping,
        kv_cache_out,
        implementation_index=implementation_index,
        stream=get_stream(key.device),
    )

    return kv_cache_out


def _ref_reshape_and_cache(key, value, kv_cache, slot_mapping, kv_cache_out):
    kv_cache_out = kv_cache_out.clone()
    slots = slot_mapping.cpu()
    block_size = kv_cache_out.size(2)

    for i in range(key.size(0)):
        slot = int(slots[i].item())

        if slot < 0:
            continue

        block_idx = slot // block_size
        offset = slot % block_size
        kv_cache_out[0, block_idx, offset, :, :] = key[i, :, :]
        kv_cache_out[1, block_idx, offset, :, :] = value[i, :, :]

    return kv_cache_out
