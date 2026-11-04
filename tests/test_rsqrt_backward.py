import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format: (shape, grad_output_strides, input_strides, grad_input_strides)
_TEST_CASES = (
    pytest.param(
        (17,),
        None,
        None,
        None,
        id="one-dimensional",
    ),
    pytest.param(
        (2, 3),
        None,
        None,
        None,
        id="two-dimensional",
    ),
    pytest.param(
        (2, 3, 4),
        None,
        None,
        None,
        id="three-dimensional",
    ),
    pytest.param(
        (2, 3, 4),
        (30, 7, 1),
        (28, 6, 1),
        (35, 8, 1),
        id="strided-tensors",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, grad_output_strides, input_strides, grad_input_strides",
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
def test_rsqrt_backward(
    shape,
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
        lambda *args: _rsqrt_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_rsqrt_backward,
        (grad_output, input, grad_input),
        {},
        rtol=rtol,
        atol=atol,
    )


def _rsqrt_backward(
    grad_output,
    input,
    grad_input,
    *,
    implementation_index,
):
    infini.ops.rsqrt_backward(
        grad_output,
        input,
        grad_input,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )

    return grad_input


def _torch_rsqrt_backward(grad_output, input, grad_input):
    differentiable_input = input.detach().clone().requires_grad_(True)
    result = torch.rsqrt(differentiable_input)
    gradient = torch.autograd.grad(result, differentiable_input, grad_output)[0]
    grad_input.copy_(gradient)

    return grad_input
