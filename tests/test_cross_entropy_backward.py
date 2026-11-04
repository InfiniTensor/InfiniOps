import infini.ops
import pytest
import torch

from tests.utils import (
    Payload,
    empty_strided,
    get_stream,
    randint_strided,
    randn_strided,
)


# Format: (input_shape, input_strides, target_strides, grad_input_strides)
# CrossEntropy uses the PyTorch convention: dimension 1 is the class dimension.
_TEST_CASES = (
    pytest.param(
        (4, 7),
        None,
        None,
        None,
        id="two-dimensional",
    ),
    pytest.param(
        (2, 3, 5),
        None,
        None,
        None,
        id="three-dimensional",
    ),
    pytest.param(
        (2, 3, 5),
        (30, 7, 1),
        None,
        (35, 8, 1),
        id="strided-input-and-output",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, input_strides, target_strides, grad_input_strides",
    _TEST_CASES,
)
@pytest.mark.parametrize("use_ignore_index", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_cross_entropy_backward(
    input_shape,
    input_strides,
    target_strides,
    grad_input_strides,
    use_ignore_index,
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
    target = randint_strided(
        0,
        input_shape[1],
        (input_shape[0],) + input_shape[2:],
        target_strides,
        dtype=torch.int64,
        device=device,
    )
    if use_ignore_index:
        target.reshape(-1)[0] = -100
    grad_output = torch.randn((), dtype=dtype, device=device)
    grad_input = empty_strided(
        input_shape,
        grad_input_strides,
        dtype=dtype,
        device=device,
    )
    grad_input.fill_(17)

    return Payload(
        lambda *args: _cross_entropy_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_cross_entropy_backward,
        (input, target, grad_output, grad_input),
        {},
        rtol=rtol,
        atol=atol,
    )


def _cross_entropy_backward(
    input,
    target,
    grad_output,
    grad_input,
    *,
    implementation_index,
):
    infini.ops.cross_entropy_backward(
        input,
        target,
        grad_output,
        grad_input,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return grad_input


def _torch_cross_entropy_backward(input, target, grad_output, grad_input):
    logits = input.detach().clone().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(
        logits,
        target,
        reduction="mean",
        ignore_index=-100,
    )
    gradient = torch.autograd.grad(loss, logits, grad_output)[0]
    grad_input.copy_(gradient)

    return grad_input
