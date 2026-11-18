import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_TEST_CASES = (
    pytest.param((17,), 2.0, False, None, None, None, id="one-dimensional-input-base"),
    pytest.param((2, 3), 3.0, False, None, None, None, id="two-dimensional-input-base"),
    pytest.param((2, 3, 4), 0.5, False, None, None, None, id="fractional-input-base"),
    pytest.param((2, 3, 4), 2.0, True, None, None, None, id="scalar-base"),
    pytest.param((2, 3, 4), 0.5, True, None, None, None, id="fractional-scalar-base"),
    pytest.param(
        (2, 3, 4), 3.0, False, (30, 7, 1), (28, 6, 1), (35, 8, 1), id="strided-tensors"
    ),
)

_DTYPE_CASES = (
    pytest.param(torch.float32, 1e-6, 1e-6, id="float32"),
    pytest.param(torch.float16, 1e-3, 1e-3, id="float16"),
    pytest.param(torch.bfloat16, 1e-2, 1e-2, id="bfloat16"),
)

_EDGE_CASES = (
    pytest.param(
        (-2.0, 0.0, 2.0),
        (1.0, -2.0, 3.0),
        0.0,
        False,
        id="zero-exponent-including-zero-input",
    ),
    pytest.param(
        (-2.0, 0.0, 2.0),
        (1.0, -2.0, 3.0),
        1.0,
        False,
        id="unit-exponent-including-zero-input",
    ),
    pytest.param(
        (-2.0, 0.0, 2.0),
        (1.0, -2.0, 3.0),
        2.0,
        False,
        id="square-negative-and-zero-input",
    ),
    pytest.param(
        (-2.0, 0.0, 2.0),
        (1.0, -2.0, 3.0),
        3.0,
        False,
        id="cube-negative-and-zero-input",
    ),
    pytest.param(
        (-2.0, -1.0, 1.0, 2.0),
        (1.0, -2.0, 0.0, 3.0),
        -2.0,
        False,
        id="negative-integer-exponent",
    ),
    pytest.param(
        (0.25, 1.0, 4.0),
        (2.0, 0.0, -2.0),
        0.5,
        False,
        id="fractional-exponent-positive-input",
    ),
    pytest.param(
        (-2.0, 0.0, 2.0),
        (1.0, -2.0, 3.0),
        1.0,
        True,
        id="unit-scalar-base-zero-derivative",
    ),
    pytest.param(
        (-2.0, 0.0, 2.0),
        (1.0, -2.0, 3.0),
        2.0,
        True,
        id="scalar-base-negative-zero-positive-exponents",
    ),
    pytest.param(
        (-2.0, 0.0, 2.0),
        (1.0, -2.0, 3.0),
        0.5,
        True,
        id="fractional-scalar-base-negative-derivative",
    ),
)

_NUMERICAL_CASES = (
    pytest.param(
        (0.5,),
        (40000.0,),
        2.0,
        False,
        id="input-base-large-grad-finite-result",
    ),
    pytest.param(
        (1.0,),
        (40000.0,),
        2.0,
        True,
        id="scalar-base-large-grad-finite-result",
    ),
    pytest.param(
        (-16.0,),
        (1.0,),
        0.5,
        True,
        id="scalar-base-pow-overflow-finite-gradient",
    ),
    pytest.param(
        (0.5,),
        (-40000.0,),
        2.0,
        False,
        id="input-base-large-negative-grad",
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
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPE_CASES)
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
    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
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


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    ("input_values", "grad_output_values", "scalar", "scalar_is_base"),
    _EDGE_CASES,
)
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPE_CASES)
def test_pow_backward_edge_cases(
    input_values,
    grad_output_values,
    scalar,
    scalar_is_base,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input = torch.tensor(input_values, dtype=dtype, device=device)
    grad_output = torch.tensor(grad_output_values, dtype=dtype, device=device)
    grad_input = torch.full_like(input, 17)

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


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    ("input_values", "grad_output_values", "scalar", "scalar_is_base"),
    _NUMERICAL_CASES,
)
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPE_CASES)
def test_pow_backward_numerical_stability(
    input_values,
    grad_output_values,
    scalar,
    scalar_is_base,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input = torch.tensor(input_values, dtype=dtype, device=device)
    grad_output = torch.tensor(grad_output_values, dtype=dtype, device=device)
    grad_input = torch.full_like(input, 17)

    def actual(*args):
        result = _pow_backward(
            *args,
            implementation_index=implementation_index,
        )
        assert torch.isfinite(result).all().item(), (
            "Expected a finite gradient, but got inf or nan"
        )
        return result

    return Payload(
        actual,
        _torch_pow_backward_high_precision,
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


def _torch_pow_backward_high_precision(
    grad_output,
    input,
    scalar,
    scalar_is_base,
    grad_input,
):
    reference_input = input.detach().to(device="cpu", dtype=torch.float64)
    reference_grad_output = grad_output.detach().to(
        device="cpu",
        dtype=torch.float64,
    )
    reference_grad_input = torch.empty_like(reference_input)
    gradient = _torch_pow_backward(
        reference_grad_output,
        reference_input,
        scalar,
        scalar_is_base,
        reference_grad_input,
    )
    expected = gradient.to(device=grad_input.device, dtype=grad_input.dtype)
    assert torch.isfinite(expected).all().item(), (
        "The regression case must have a finite expected gradient"
    )
    grad_input.copy_(expected)
    return grad_input
