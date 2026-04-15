import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_npu_stream, rand_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, x_strides, out_strides",
    (
        ((13, 8), None, None),
        ((16, 11264), None, None),
        ((4, 4, 11264), None, None),
        ((1, 8), None, None),
        ((32, 5632), None, None),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_silu_and_mul(shape, x_strides, out_strides, dtype, device, rtol, atol):
    x = rand_strided(shape, x_strides, dtype=dtype, device=device)
    d = shape[-1] // 2
    out_shape = (*shape[:-1], d)
    out = empty_strided(out_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        _silu_and_mul,
        _torch_silu_and_mul,
        (x, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _silu_and_mul(x, out):
    if x.device.type == "npu":
        infini.ops.silu_and_mul(
            x,
            -1,
            out,
            stream=get_npu_stream(x),
        )
    else:
        infini.ops.silu_and_mul(x, -1, out)

    return out


def _torch_silu_and_mul(x, out):
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    result = up * torch.sigmoid(gate) * gate

    return result.to(out.dtype)
