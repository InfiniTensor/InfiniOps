import infini.ops
import pytest
import torch

from tests.utils import get_stream, randn_strided


@pytest.mark.parametrize("padded_source", (False, True))
@pytest.mark.parametrize(
    "kv_cache_dtype",
    ("auto", "fp8", "fp8_e4m3", "fp8_e5m2"),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-3, 1e-3),
    ),
)
def test_reshape_and_cache(
    padded_source,
    kv_cache_dtype,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    num_tokens, num_heads, head_size = 4, 2, 32
    num_blocks, block_size = 3, 4
    source_strides = (
        (num_heads * head_size + 16, head_size, 1) if padded_source else None
    )
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
    quantized = kv_cache_dtype != "auto"
    cache_dtype = torch.uint8 if quantized else dtype
    x = 16 if quantized else 16 // key.element_size()
    key_cache = torch.zeros(
        (num_blocks, num_heads, head_size // x, block_size, x),
        dtype=cache_dtype,
        device=device,
    )
    value_cache = torch.zeros(
        (num_blocks, num_heads, head_size, block_size),
        dtype=cache_dtype,
        device=device,
    )
    k_scale = torch.tensor((0.5,), dtype=torch.float32, device=device)
    v_scale = torch.tensor((0.25,), dtype=torch.float32, device=device)
    expected_key_cache = key_cache.clone()
    expected_value_cache = value_cache.clone()

    _torch_reshape_and_cache(
        key,
        value,
        expected_key_cache,
        expected_value_cache,
        slot_mapping,
        k_scale,
        v_scale,
        kv_cache_dtype,
    )
    result = infini.ops.reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
        kv_cache_dtype,
        implementation_index=implementation_index,
        stream=get_stream(key.device),
    )

    assert result is None
    torch.testing.assert_close(key_cache, expected_key_cache, rtol=rtol, atol=atol)
    torch.testing.assert_close(value_cache, expected_value_cache, rtol=rtol, atol=atol)


def _torch_reshape_and_cache(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    k_scale,
    v_scale,
    kv_cache_dtype,
):
    token_indices = torch.nonzero(slot_mapping >= 0).flatten()
    slots = slot_mapping[token_indices]
    block_size = key_cache.shape[3]
    block_indices = torch.div(slots, block_size, rounding_mode="floor")
    block_offsets = torch.remainder(slots, block_size)
    x = key_cache.shape[4]
    key_values = key[token_indices].view(
        token_indices.numel(), key.shape[1], key.shape[2] // x, x
    )
    value_values = value[token_indices]

    if kv_cache_dtype != "auto":
        fp8_dtype = (
            torch.float8_e5m2 if kv_cache_dtype == "fp8_e5m2" else torch.float8_e4m3fn
        )
        key_values = (key_values / k_scale).to(fp8_dtype).view(torch.uint8)
        value_values = (value_values / v_scale).to(fp8_dtype).view(torch.uint8)

    key_cache[block_indices, :, :, block_offsets, :] = key_values
    value_cache[block_indices, :, :, block_offsets] = value_values


@pytest.mark.smoke
def test_aten_external_stream(device, implementation_index):
    if implementation_index != 1 or device == "cpu":
        pytest.skip("Requires the accelerator ATen provider")
    accelerator = getattr(torch, torch.device(device).type)
    previous = get_stream(device)
    stream = accelerator.Stream()
    raw = getattr(stream, f"{torch.device(device).type}_stream")
    x = torch.arange(8, device=device, dtype=torch.float32).reshape(1, 1, 8)
    key_cache = torch.zeros((1, 1, 2, 2, 4), device=device)
    out = torch.zeros((1, 1, 8, 2), device=device)
    indices = torch.zeros((1,), dtype=torch.int64, device=device)
    scale = torch.ones((1,), device=device)
    reference = out.clone()
    reference[0, 0, :, 0] = x.flatten()
    initial = out.clone()
    stream.wait_stream(accelerator.current_stream())

    def invoke():
        infini.ops.reshape_and_cache(
            x,
            x,
            key_cache,
            out,
            indices,
            scale,
            scale,
            "auto",
            implementation_index=implementation_index,
            stream=raw,
        )

    try:
        invoke()
        accelerator.synchronize()
        out.copy_(initial)
        accelerator.synchronize()
        with accelerator.stream(stream):
            snapshot = torch.empty_like(out)
            snapshot.copy_(out)
        stream.synchronize()
        delayed = None
        if device == "cuda":
            accelerator._sleep(200_000_000)
            delayed = accelerator.Event()
            delayed.record()
        invoke()
        with accelerator.stream(stream):
            snapshot.copy_(out)
        stream.synchronize()
        if delayed is not None:
            assert not delayed.query(), (
                "The unrelated stream completed before the snapshot"
            )
        assert get_stream(device) == previous
    finally:
        accelerator.synchronize()
    torch.testing.assert_close(snapshot, reference, rtol=1e-4, atol=1e-5)
