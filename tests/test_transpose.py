import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format: (input_shape, input_strides, dim0, dim1, out_strides)
_TEST_CASES = (
    pytest.param(
        (2, 3),
        None,
        0,
        1,
        None,
        id="two-dimensional",
    ),
    pytest.param(
        (2, 3, 4),
        None,
        0,
        2,
        None,
        id="three-dimensional-outer-dimensions",
    ),
    pytest.param(
        (2, 3, 4),
        None,
        -1,
        -2,
        None,
        id="negative-dimensions",
    ),
    pytest.param(
        (2, 3, 4),
        None,
        1,
        1,
        None,
        id="same-dimension",
    ),
    pytest.param(
        (2, 3, 4),
        (20, 6, 1),
        0,
        2,
        (10, 3, 1),
        id="strided-input-and-output",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, input_strides, dim0, dim1, out_strides",
    _TEST_CASES,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 0, 0),
        (torch.float16, 0, 0),
        (torch.bfloat16, 0, 0),
    ),
)
def test_transpose(
    input_shape,
    input_strides,
    dim0,
    dim1,
    out_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(
        input_shape,
        input_strides,
        dtype=dtype,
        device=device,
    )
    out_shape = list(input_shape)
    normalized_dim0 = dim0 % len(input_shape)
    normalized_dim1 = dim1 % len(input_shape)
    out_shape[normalized_dim0], out_shape[normalized_dim1] = (
        out_shape[normalized_dim1],
        out_shape[normalized_dim0],
    )
    out = empty_strided(
        tuple(out_shape),
        out_strides,
        dtype=dtype,
        device=device,
    )
    out.fill_(17)

    return Payload(
        lambda *args: _transpose(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_transpose,
        (input, dim0, dim1, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _transpose(input, dim0, dim1, out, *, implementation_index):
    infini.ops.transpose(
        input,
        dim0,
        dim1,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_transpose(input, dim0, dim1, out):
    out.copy_(torch.transpose(input, dim0, dim1))

    return out
