from contextlib import nullcontext

import infini.ops
import pytest
import torch

from tests.utils import get_stream


_TEST_CASES = (
    (4, 4, "none", 0.0, False, None, False),
    (3, 5, "float", 0.0, False, 0.5, False),
    (4, 4, "bool", 0.0, False, None, False),
    (4, 4, "none", 0.0, True, None, False),
    (3, 5, "none", 0.0, False, None, True),
    (4, 4, "none", 0.2, False, None, False),
)


@pytest.mark.parametrize(
    "query_length, key_length, mask_kind, dropout_p, is_causal, scale, enable_gqa",
    _TEST_CASES,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 2e-2, 1e-2),
    ),
)
def test_scaled_dot_product_attention(
    query_length,
    key_length,
    mask_kind,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    batch_size, query_heads, head_size, value_size = 2, 4, 8, 6
    kv_heads = 2 if enable_gqa else query_heads
    query = torch.randn(
        batch_size,
        query_heads,
        query_length,
        head_size,
        dtype=dtype,
        device=device,
    )
    key = torch.randn(
        batch_size,
        kv_heads,
        key_length,
        head_size,
        dtype=dtype,
        device=device,
    )
    value = torch.randn(
        batch_size,
        kv_heads,
        key_length,
        value_size,
        dtype=dtype,
        device=device,
    )
    attn_mask = _make_mask(mask_kind, query_length, key_length, dtype, device)
    out = torch.empty(
        batch_size,
        query_heads,
        query_length,
        value_size,
        dtype=dtype,
        device=device,
    )

    torch.manual_seed(1234)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    torch.manual_seed(1234)
    if (
        attn_mask is None
        and dropout_p == 0.0
        and not is_causal
        and scale is None
        and not enable_gqa
    ):
        result = infini.ops.scaled_dot_product_attention(
            query,
            key,
            value,
            out,
            implementation_index=implementation_index,
            stream=get_stream(query.device),
        )
    else:
        result = infini.ops.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            out,
            implementation_index=implementation_index,
            stream=get_stream(query.device),
        )

    assert result is None
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


def _make_mask(mask_kind, query_length, key_length, dtype, device):
    if mask_kind == "none":
        return None

    if mask_kind == "bool":
        mask = torch.ones(query_length, key_length, dtype=torch.bool, device=device)
        mask[0, -1] = False

        return mask

    mask = torch.zeros(query_length, key_length, dtype=dtype, device=device)
    mask[0, -1] = -3.0

    return mask


@pytest.mark.smoke
@pytest.mark.parametrize("managed", [False, True])
def test_aten_external_stream(device, implementation_index, managed):
    if implementation_index != 1 or device == "cpu":
        pytest.skip("Requires the accelerator ATen provider")
    accelerator = getattr(torch, torch.device(device).type)
    previous = get_stream(device)
    stream = accelerator.Stream()
    raw = getattr(stream, f"{torch.device(device).type}_stream")
    x = torch.randn((1, 2, 4, 8), device=device)
    out = torch.full_like(x, -123)
    reference = torch.nn.functional.scaled_dot_product_attention(x, x, x)
    initial = out.clone()
    stream.wait_stream(accelerator.current_stream())

    def invoke():
        with accelerator.stream(stream) if managed else nullcontext():
            infini.ops.scaled_dot_product_attention(
                x, x, x, out, implementation_index=implementation_index, stream=raw
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
    except RuntimeError as error:
        # Preserve the SDK's explicit rejection of an event on a raw stream.
        if (
            device != "npu"
            or managed
            or "External NPU stream is not supported in NPUEvent::record"
            not in str(error)
        ):
            raise
        assert get_stream(device) == previous
        assert torch.all(out == initial)
        return
    finally:
        accelerator.synchronize()
    torch.testing.assert_close(snapshot, reference, rtol=1e-4, atol=1e-5)
