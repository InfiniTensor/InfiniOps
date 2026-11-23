import decimal

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
    differentiable_input = (
        input.detach().to(device="cpu", dtype=torch.float64).requires_grad_(True)
    )
    result = torch.rsqrt(differentiable_input)
    upstream = grad_output.detach().to(device="cpu", dtype=torch.float64)
    gradient = torch.autograd.grad(result, differentiable_input, upstream)[0]
    grad_input.copy_(gradient)

    return grad_input


@pytest.mark.parametrize(
    "dtype,value,upstream",
    (
        (torch.float16, 2**-12, 0.001),
        (torch.float16, 2**-24, 2**-24),
        (torch.bfloat16, 2**-120, 2**-100),
        (torch.float32, 2**-100, 2**-80),
        (torch.float64, 2**-800, 2**-700),
    ),
)
def test_rsqrt_backward_finite_intermediates(
    dtype, value, upstream, device, implementation_index
):
    input = torch.full((2, 3), value, dtype=dtype, device=device)
    grad_output = torch.full_like(input, upstream)
    backing = torch.full((32,), 17, dtype=dtype, device=device)
    output = backing.as_strided((2, 3), (6, 2), 2)

    with decimal.localcontext() as context:
        context.prec = 80
        x = decimal.Decimal.from_float(input[0, 0].item())
        g = decimal.Decimal.from_float(grad_output[0, 0].item())
        expected_value = float(-g / (2 * x * x.sqrt()))

    expected = backing.clone()
    expected.as_strided((2, 3), (6, 2), 2).fill_(expected_value)
    _rsqrt_backward(
        grad_output, input, output, implementation_index=implementation_index
    )
    assert torch.isfinite(output).all().item()
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)
