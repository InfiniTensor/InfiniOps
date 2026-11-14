import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format: (input_shape, num_inputs, dim, grad_output_strides, grad_input_strides)
_TEST_CASES = (
    pytest.param(
        (2, 5),
        3,
        0,
        None,
        None,
        id="two-dimensional-first-dimension",
    ),
    pytest.param(
        (2, 5),
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
        (2, 3),
        3,
        1,
        (20, 5, 1),
        (5, 1),
        id="strided-gradients",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "input_shape",
        "num_inputs",
        "dim",
        "grad_output_strides",
        "grad_input_strides",
    ),
    _TEST_CASES,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_stack_backward(
    input_shape,
    num_inputs,
    dim,
    grad_output_strides,
    grad_input_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    normalized_dim = dim % len(input_shape + (0,))
    grad_output_shape = (
        input_shape[:normalized_dim] + (num_inputs,) + input_shape[normalized_dim:]
    )
    grad_output = randn_strided(
        grad_output_shape,
        grad_output_strides,
        dtype=dtype,
        device=device,
    )
    grad_inputs = [
        empty_strided(
            input_shape,
            grad_input_strides,
            dtype=dtype,
            device=device,
        )
        for _ in range(num_inputs)
    ]
    for grad_input in grad_inputs:
        grad_input.fill_(17)

    return Payload(
        lambda *args: _stack_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_stack_backward,
        (grad_output, dim, grad_inputs),
        {},
        rtol=rtol,
        atol=atol,
    )


def _stack_backward(
    grad_output,
    dim,
    grad_inputs,
    *,
    implementation_index,
):
    infini.ops.stack_backward(
        grad_output,
        dim,
        grad_inputs,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )

    return torch.stack(grad_inputs, dim=dim)


def _torch_stack_backward(grad_output, dim, grad_inputs):
    for grad_input, expected in zip(grad_inputs, torch.unbind(grad_output, dim=dim)):
        grad_input.copy_(expected)

    return torch.stack(grad_inputs, dim=dim)
