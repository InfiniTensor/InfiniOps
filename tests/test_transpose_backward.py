import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format: (grad_input_shape, dim0, dim1, grad_output_strides, grad_input_strides)
_TEST_CASES = (
    pytest.param(
        (2, 3),
        0,
        1,
        None,
        None,
        id="two-dimensional",
    ),
    pytest.param(
        (2, 3, 4),
        0,
        2,
        None,
        None,
        id="three-dimensional-outer-dimensions",
    ),
    pytest.param(
        (2, 3, 4),
        -1,
        -2,
        None,
        None,
        id="negative-dimensions",
    ),
    pytest.param(
        (2, 3, 4),
        1,
        1,
        None,
        None,
        id="same-dimension",
    ),
    pytest.param(
        (2, 3, 4),
        0,
        2,
        (10, 3, 1),
        (20, 6, 1),
        id="strided-gradients",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "grad_input_shape",
        "dim0",
        "dim1",
        "grad_output_strides",
        "grad_input_strides",
    ),
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
def test_transpose_backward(
    grad_input_shape,
    dim0,
    dim1,
    grad_output_strides,
    grad_input_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    grad_output_shape = list(grad_input_shape)
    normalized_dim0 = dim0 % len(grad_input_shape)
    normalized_dim1 = dim1 % len(grad_input_shape)
    grad_output_shape[normalized_dim0], grad_output_shape[normalized_dim1] = (
        grad_output_shape[normalized_dim1],
        grad_output_shape[normalized_dim0],
    )
    grad_output = randn_strided(
        tuple(grad_output_shape),
        grad_output_strides,
        dtype=dtype,
        device=device,
    )
    grad_input = empty_strided(
        grad_input_shape,
        grad_input_strides,
        dtype=dtype,
        device=device,
    )
    grad_input.fill_(17)

    return Payload(
        lambda *args: _transpose_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_transpose_backward,
        (grad_output, dim0, dim1, grad_input),
        {},
        rtol=rtol,
        atol=atol,
    )


def _transpose_backward(
    grad_output,
    dim0,
    dim1,
    grad_input,
    *,
    implementation_index,
):
    infini.ops.transpose_backward(
        grad_output,
        dim0,
        dim1,
        grad_input,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )

    return grad_input


def _torch_transpose_backward(grad_output, dim0, dim1, grad_input):
    grad_input.copy_(torch.transpose(grad_output, dim0, dim1))

    return grad_input
