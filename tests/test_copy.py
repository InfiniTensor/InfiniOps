import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, out_shape, input_strides, out_strides",
    (
        ((100, 100), (100, 100), (1, 100), (100, 1)),
        ((2, 2, 2, 4), (2, 2, 2, 4), (16, 8, 4, 1), (16, 8, 1, 2)),
        ((8, 4, 20, 64), (8, 4, 20, 64), (5120, 64, 256, 1), None),
        ((1, 64), (15, 64), None, None),
        ((64,), (8, 4, 64), None, None),
    ),
)
@pytest.mark.parametrize("non_blocking", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 0, 0),
        (torch.float16, 0, 0),
        (torch.bfloat16, 0, 0),
    ),
)
def test_copy(
    input_shape,
    out_shape,
    input_strides,
    out_strides,
    non_blocking,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(input_shape, input_strides, dtype=dtype, device=device)
    out = empty_strided(out_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        _copy,
        _torch_copy,
        (input, non_blocking, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _copy(input, non_blocking, out):
    infini.ops.copy(input, non_blocking, out, stream=get_stream(input.device))

    return out


def _torch_copy(input, non_blocking, out):
    out.copy_(input, non_blocking=non_blocking)

    return out
