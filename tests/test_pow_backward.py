import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format: (shape, scalar, scalar_is_base, grad_output_strides, input_strides,
#          grad_input_strides)
_TEST_CASES = (
    pytest.param(
        (17,),
        2.0,
        False,
        None,
        None,
        None,
        id="one-dimensional-input-base",
    ),
    pytest.param(
        (2, 3),
        3.0,
        False,
        None,
        None,
        None,
        id="two-dimensional-input-base",
    ),
    pytest.param(
        (2, 3, 4),
        0.5,
        False,
        None,
        None,
        None,
        id="fractional-input-base",
    ),
    pytest.param(
        (2, 3, 4),
        2.0,
        True,
        None,
        None,
        None,
        id="scalar-base",
    ),
    pytest.param(
        (2, 3, 4),
        0.5,
        True,
        None,
        None,
        None,
        id="fractional-scalar-base",
    ),
    pytest.param(
        (2, 3, 4),
        3.0,
        False,
        (30, 7, 1),
        (28, 6, 1),
        (35, 8, 1),
        id="strided-tensors",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "shape",
        "scalar",
        "scalar_is_base",
        "grad_output_strides",
        "input_strides",
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
def test_pow_backward(
    shape,
    scalar,
    scalar_is_base,
    grad_output_strides,
    input_strides,
    grad_input_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    grad_output = randn_strided(
        shape,
        grad_output_strides,
        dtype=dtype,
        device=device,
    )
    input = randn_strided(
        shape,
        input_strides,
        dtype=dtype,
        device=device,
    )
    input.abs_().add_(0.5)
    grad_input = empty_strided(
        shape,
        grad_input_strides,
        dtype=dtype,
        device=device,
    )
    grad_input.fill_(17)

    return Payload(
        lambda *args: _pow_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_pow_backward,
        (grad_output, input, scalar, scalar_is_base, grad_input),
        {},
        rtol=rtol,
        atol=atol,
    )


def _pow_backward(
    grad_output,
    input,
    scalar,
    scalar_is_base,
    grad_input,
    *,
    implementation_index,
):
    infini.ops.pow_backward(
        grad_output,
        input,
        scalar,
        scalar_is_base,
        grad_input,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )

    return grad_input


def _torch_pow_backward(
    grad_output,
    input,
    scalar,
    scalar_is_base,
    grad_input,
):
    differentiable_input = input.detach().clone().requires_grad_(True)

    if scalar_is_base:
        result = torch.pow(scalar, differentiable_input)
    else:
        result = torch.pow(differentiable_input, scalar)

    gradient = torch.autograd.grad(result, differentiable_input, grad_output)[0]
    grad_input.copy_(gradient)

    return grad_input
