import infini.ops
import pytest
import torch

from tests.utils import (
    Payload,
    empty_strided,
    get_stream,
    randint_strided,
    randn_strided,
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides, inplace",
    (
        ((13, 4), None, None, False),
        ((13, 4), None, None, True),
        ((13, 4), (10, 1), (10, 1), False),
        ((13, 4), (0, 1), None, False),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), False),
        ((0, 4), None, None, False),
    ),
)
@pytest.mark.parametrize(
    "dtype, value",
    (
        (torch.uint8, 3.0),
        (torch.int8, -3.0),
        (torch.int16, -7.0),
        (torch.int32, 11.0),
        (torch.int64, -13.0),
        (torch.float64, -1.25),
        (torch.float32, 2.5),
        (torch.float16, -3.5),
        (torch.bfloat16, 4.5),
    ),
)
def test_fill(
    shape,
    input_strides,
    out_strides,
    inplace,
    dtype,
    value,
    device,
):
    if device in ("mlu", "musa") and dtype == torch.float64:
        pytest.skip(f"{device.upper()} does not support float64 fill")

    input = _make_input(shape, input_strides, dtype=dtype, device=device)
    out = (
        input
        if inplace
        else empty_strided(shape, out_strides, dtype=dtype, device=device)
    )

    return Payload(_fill, _torch_fill, (input, value, out), {}, rtol=0, atol=0)


def _make_input(shape, strides, *, dtype, device):
    if dtype.is_floating_point:
        return randn_strided(shape, strides, dtype=dtype, device=device)

    return randint_strided(1, 16, shape, strides, dtype=dtype, device=device)


def _fill(input, value, out):
    infini.ops.fill(input, value, out, stream=get_stream(input.device))

    return out


def _torch_fill(input, value, out):
    out.fill_(value)

    return out
