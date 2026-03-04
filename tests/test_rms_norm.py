import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, weight_shape, input_strides, weight_strides, out_strides",
    (
        ((1, 64), (64,), None, None, None),
        ((2, 128), (128,), None, None, None),
        ((4, 48, 64), (64,), None, None, None),
        ((2, 4, 2048), (2048,), None, None, None),
        ((1, 64), (64,), (64, 1), (1,), (64, 1)),
        ((4, 48, 64), (64,), (3072, 64, 1), (1,), (3072, 64, 1)),
    ),
)
@pytest.mark.parametrize("eps", (1e-6, 1e-5))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_rms_norm(
    input_shape,
    weight_shape,
    input_strides,
    weight_strides,
    out_strides,
    eps,
    dtype,
    device,
    rtol,
    atol,
):
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("CPU backend does not support fp16/bf16")

    input = randn_strided(input_shape, input_strides, dtype=dtype, device=device)
    weight = randn_strided(weight_shape, weight_strides, dtype=dtype, device=device)
    out = empty_strided(input_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        _rms_norm,
        _torch_rms_norm,
        (input, weight),
        {"eps": eps, "out": out},
        rtol=rtol,
        atol=atol,
    )


def _rms_norm(input, weight, *, eps=1e-6, out=None):
    infini.ops.rms_norm(input, weight, eps, out)

    return out


def _torch_rms_norm(input, weight, *, eps=1e-6, out=None):
    return torch.nn.functional.rms_norm(input, input.shape[-1:], weight=weight, eps=eps)
