import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, gate_strides, out_strides",
    (
        ((13, 4), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((13, 4, 4), None, None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
        ((16, 5632), None, None, None),
        ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
        ((4, 4, 5632), None, None, None),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_swiglu(
    shape, input_strides, gate_strides, out_strides, dtype, device, rtol, atol
):
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("CPU backend does not support fp16/bf16")
    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    gate = randn_strided(shape, gate_strides, dtype=dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        _swiglu, _torch_swiglu, (input, gate, out), {}, rtol=rtol, atol=atol
    )


def _swiglu(input, gate, out):
    infini.ops.swiglu(input, gate, out)

    return out


def _torch_swiglu(input, gate, out):
    # PyTorch implementation of SwiGLU
    # SwiGLU(x, gate) = Swish(x) * gate
    # where Swish(x) = x * sigmoid(x)
    swish_x = input * torch.sigmoid(input)
    return torch.mul(swish_x, gate, out=out)
