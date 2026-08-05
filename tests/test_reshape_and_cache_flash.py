import infini.ops
import pytest
import torch

from tests.utils import get_stream, randn_strided


@pytest.mark.parametrize("per_head_scale", (False, True))
@pytest.mark.parametrize("cache_layout", ("NHD", "HND"))
@pytest.mark.parametrize("padded_source", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_reshape_and_cache_flash(
    per_head_scale,
    cache_layout,
    padded_source,
    dtype,
    implementation_index,
    device,
):
    num_tokens, num_heads, head_size = 5, 2, 16
    num_blocks, block_size = 3, 4
    head_stride = head_size + 3 if padded_source else head_size
    token_stride = num_heads * head_stride + 5 if padded_source else None
    source_strides = (token_stride, head_stride, 1) if padded_source else None
    key = randn_strided(
        (num_tokens, num_heads, head_size),
        source_strides,
        dtype=dtype,
        device=device,
    )
    value = randn_strided(
        (num_tokens, num_heads, head_size),
        source_strides,
        dtype=dtype,
        device=device,
    )
    slot_mapping = torch.tensor((0, -1, 5, 10), dtype=torch.int64, device=device)
    key_cache = _make_cache(
        num_blocks, block_size, num_heads, head_size, cache_layout, dtype, device
    )
    value_cache = _make_cache(
        num_blocks, block_size, num_heads, head_size, cache_layout, dtype, device
    )
    scale_shape = (num_heads,) if per_head_scale else ()
    k_scale = torch.ones(scale_shape, dtype=torch.float32, device=device)
    v_scale = torch.ones(scale_shape, dtype=torch.float32, device=device)
    expected_key_cache = key_cache.clone(memory_format=torch.preserve_format)
    expected_value_cache = value_cache.clone(memory_format=torch.preserve_format)

    for token_idx, slot in enumerate(slot_mapping.cpu().tolist()):
        if slot < 0:
            continue

        block_idx, block_offset = divmod(slot, block_size)
        expected_key_cache[block_idx, block_offset] = key[token_idx]
        expected_value_cache[block_idx, block_offset] = value[token_idx]

    result = infini.ops.reshape_and_cache_flash(
        key,
        value,
        slot_mapping,
        k_scale,
        v_scale,
        "auto",
        key_cache,
        value_cache,
        implementation_index=implementation_index,
        stream=get_stream(key.device),
    )

    assert result is None
    torch.testing.assert_close(key_cache, expected_key_cache)
    torch.testing.assert_close(value_cache, expected_value_cache)


def _make_cache(num_blocks, block_size, num_heads, head_size, layout, dtype, device):
    if layout == "NHD":
        shape = (num_blocks, block_size, num_heads, head_size)
        return torch.randn(shape, dtype=dtype, device=device)

    physical_shape = (num_blocks, num_heads, block_size, head_size)
    return torch.randn(physical_shape, dtype=dtype, device=device).permute(0, 2, 1, 3)
