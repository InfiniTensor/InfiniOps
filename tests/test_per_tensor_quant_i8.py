"""Metax unit tests for the per-tensor INT8 quantization operators.

Covers the metax implementations in `per_tensor_i8`:
- PerTensorQuantI8: dynamic (scale = absmax / 127) and static (caller
  provided scale) quantization, round-half-away-from-zero, saturation
  to [-127, 127].
- PerTensorDequantI8: packed int8 * scale -> floating point.
- Quantize -> dequantize round-trip within 1 ulp of the step size.
"""

import torch
import pytest

import infini.ops

from tests.utils import get_stream


if not hasattr(infini.ops, "per_tensor_quant_i8"):
    pytest.skip(
        "`per_tensor_quant_i8` is not available on this platform",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "dtype",
    (torch.float32, torch.float16, torch.bfloat16),
)
def test_per_tensor_quant_i8_dynamic(dtype, device):
    x = (torch.randn(3, 5) * 4.0).to(dtype).to(device)
    packed = torch.empty(x.shape, dtype=torch.int8, device=device)
    scale = torch.empty(1, dtype=torch.float32, device=device)

    infini.ops.per_tensor_quant_i8(
        x, packed, scale, None, False, stream=get_stream(device)
    )

    expected_scale = x.abs().max().float() / 127.0
    torch.testing.assert_close(scale[0], expected_scale, rtol=1e-5, atol=1e-8)
    q = (x.float() / expected_scale).round().clamp(-127, 127).to(torch.int8)
    torch.testing.assert_close(packed, q, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype",
    (torch.float32, torch.float16),
)
def test_per_tensor_quant_i8_static(dtype, device):
    x = (torch.randn(2, 3, 4) * 2.0).to(dtype).to(device)
    packed = torch.empty(x.shape, dtype=torch.int8, device=device)
    scale = torch.tensor([0.05], dtype=torch.float32, device=device)

    infini.ops.per_tensor_quant_i8(
        x, packed, scale, None, True, stream=get_stream(device)
    )

    q = (x.float() / 0.05).round().clamp(-127, 127).to(torch.int8)
    torch.testing.assert_close(packed, q, rtol=0, atol=0)


def test_per_tensor_quant_i8_saturation(device):
    """Values beyond scale*127 saturate instead of wrapping."""
    x = torch.tensor((200.0, -200.0, 1.0), dtype=torch.float32, device=device)
    packed = torch.empty(x.shape, dtype=torch.int8, device=device)
    scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    infini.ops.per_tensor_quant_i8(
        x, packed, scale, None, True, stream=get_stream(device)
    )

    assert packed[0].item() == 127
    assert packed[1].item() == -127
    assert packed[2].item() == 1


@pytest.mark.parametrize(
    "dtype",
    (torch.float32, torch.float16, torch.bfloat16),
)
def test_per_tensor_quant_dequant_roundtrip(dtype, device):
    x = (torch.randn(4, 6) * 3.0).to(dtype).to(device)
    packed = torch.empty(x.shape, dtype=torch.int8, device=device)
    scale = torch.empty(1, dtype=torch.float32, device=device)
    restored = torch.empty(x.shape, dtype=dtype, device=device)

    infini.ops.per_tensor_quant_i8(
        x, packed, scale, None, False, stream=get_stream(device)
    )
    infini.ops.per_tensor_dequant_i8(
        restored, packed, scale, None, stream=get_stream(device)
    )

    step = scale[0].float()
    torch.testing.assert_close(restored.float(), x.float(), rtol=0, atol=step)
