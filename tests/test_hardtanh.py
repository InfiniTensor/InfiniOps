import pytest
import torch
import torch.nn.functional as F

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
def test_hardtanh(
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
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda input, out: _hardtanh(input, out, implementation_index),
        _torch_hardtanh,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _hardtanh(input, out, implementation_index):
    infini.ops.hardtanh(
        input,
        -0.5,
        0.5,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_hardtanh(input, out):
    out.copy_(F.hardtanh(input, min_val=-0.5, max_val=0.5))

    return out
