import pytest
import torch

import infini.ops

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_SHAPE_CASES = (
    ((13, 4), None, None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1), (10, 1)),
    ((13, 4, 4), None, None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((16, 5632), None, None, None, None),
    ((4, 4, 5632), None, None, None, None),
)

_FLOAT_DTYPE_CASES = (
    (torch.float32, 1e-6, 1e-6),
    (torch.float16, 1e-3, 1e-3),
    (torch.bfloat16, 1e-2, 5e-3),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, tensor1_strides, tensor2_strides, out_strides",
    _SHAPE_CASES,
)
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _FLOAT_DTYPE_CASES)
def test_addcmul(
    shape,
    input_strides,
    tensor1_strides,
    tensor2_strides,
    out_strides,
    dtype,
    device,
    implementation_index,
    rtol,
    atol,
):
    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    tensor1 = randn_strided(shape, tensor1_strides, dtype=dtype, device=device)
    tensor2 = randn_strided(shape, tensor2_strides, dtype=dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda input, tensor1, tensor2, out: _addcmul(
            input, tensor1, tensor2, out, implementation_index
        ),
        _torch_addcmul,
        (input, tensor1, tensor2, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _addcmul(input, tensor1, tensor2, out, implementation_index):
    infini.ops.addcmul(
        input,
        tensor1,
        tensor2,
        0.5,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_addcmul(input, tensor1, tensor2, out):
    out.copy_(torch.addcmul(input, tensor1, tensor2, value=0.5))

    return out
