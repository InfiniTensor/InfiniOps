import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format: (input_shape, repeats, dim, grad_output_strides, grad_input_strides)
_TEST_CASES = (
    pytest.param(
        (5,),
        3,
        0,
        None,
        None,
        id="one-dimensional",
    ),
    pytest.param(
        (2, 3),
        2,
        0,
        None,
        None,
        id="two-dimensional-first-dimension",
    ),
    pytest.param(
        (2, 3),
        4,
        1,
        None,
        None,
        id="two-dimensional-second-dimension",
    ),
    pytest.param(
        (2, 3, 4),
        2,
        -1,
        None,
        None,
        id="three-dimensional-negative-dimension",
    ),
    pytest.param(
        (2, 3, 4),
        1,
        1,
        None,
        None,
        id="repeat-once",
    ),
    pytest.param(
        (2, 3, 4),
        2,
        1,
        (40, 8, 1),
        (20, 6, 1),
        id="strided-gradients",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, repeats, dim, grad_output_strides, grad_input_strides",
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
def test_repeat_interleave_backward(
    input_shape,
    repeats,
    dim,
    grad_output_strides,
    grad_input_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    normalized_dim = dim % len(input_shape)
    grad_output_shape = list(input_shape)
    grad_output_shape[normalized_dim] *= repeats

    grad_output = randn_strided(
        tuple(grad_output_shape),
        grad_output_strides,
        dtype=dtype,
        device=device,
    )
    grad_input = empty_strided(
        input_shape,
        grad_input_strides,
        dtype=dtype,
        device=device,
    )
    grad_input.fill_(17)

    return Payload(
        lambda *args: _repeat_interleave_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_repeat_interleave_backward,
        (grad_output, dim, grad_input),
        {},
        rtol=rtol,
        atol=atol,
    )


def _repeat_interleave_backward(
    grad_output,
    dim,
    grad_input,
    *,
    implementation_index,
):
    infini.ops.repeat_interleave_backward(
        grad_output,
        dim,
        grad_input,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )

    return grad_input


def _torch_repeat_interleave_backward(grad_output, dim, grad_input):
    dim = dim % grad_output.ndim
    repeats = grad_output.shape[dim] // grad_input.shape[dim]
    reshaped_shape = list(grad_output.shape)
    reshaped_shape[dim] = grad_input.shape[dim]
    reshaped_shape.insert(dim + 1, repeats)
    grad_input.copy_(grad_output.reshape(reshaped_shape).sum(dim=dim + 1))

    return grad_input
