import pytest
import torch

import infini.ops

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_SHAPE_CASES = (
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((13, 4, 4), None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
    ((16, 5632), None, None),
    ((4, 4, 5632), None, None),
)

_FLOAT_DTYPE_CASES = (
    (torch.float32, 0.0, 0.0),
    (torch.float16, 0.0, 0.0),
    (torch.bfloat16, 0.0, 0.0),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("shape, input_strides, out_strides", _SHAPE_CASES)
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _FLOAT_DTYPE_CASES)
def test_nan_to_num(
    shape,
    input_strides,
    out_strides,
    dtype,
    device,
    implementation_index,
    rtol,
    atol,
):
    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    _inject_non_finite(input)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda input, out: _nan_to_num(input, out, implementation_index),
        _torch_nan_to_num,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _inject_non_finite(input):
    flat = input.as_strided(
        (input.untyped_storage().size() // input.element_size(),), (1,)
    )

    flat[0] = float("nan")
    flat[1] = float("inf")
    flat[2] = float("-inf")


def _nan_to_num(input, out, implementation_index):
    infini.ops.nan_to_num(
        input,
        0.0,
        1.0,
        -1.0,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_nan_to_num(input, out):
    out.copy_(torch.nan_to_num(input, nan=0.0, posinf=1.0, neginf=-1.0))

    return out
