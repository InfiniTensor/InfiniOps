import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format: (input_shape, dim, keep_dim, grad_output_strides, grad_input_strides)
_TEST_CASES = (
    pytest.param(
        (5,),
        0,
        False,
        None,
        None,
        id="one-dimensional-no-keep-dim",
    ),
    pytest.param(
        (2, 3),
        1,
        False,
        None,
        None,
        id="two-dimensional-no-keep-dim",
    ),
    pytest.param(
        (2, 3),
        1,
        True,
        None,
        None,
        id="two-dimensional-keep-dim",
    ),
    pytest.param(
        (2, 3, 4),
        -1,
        False,
        None,
        None,
        id="three-dimensional-negative-dimension",
    ),
    pytest.param(
        (2, 3, 4),
        0,
        True,
        None,
        None,
        id="three-dimensional-first-dimension-keep-dim",
    ),
    pytest.param(
        (2, 3, 4),
        1,
        False,
        (10, 1),
        (24, 7, 1),
        id="strided-gradients",
    ),
    pytest.param(
        (2, 3, 4),
        1,
        True,
        (10, 1, 1),
        (24, 7, 1),
        id="strided-gradients-keep-dim",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, dim, keep_dim, grad_output_strides, grad_input_strides",
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
def test_mean_backward(
    input_shape,
    dim,
    keep_dim,
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
    if keep_dim:
        grad_output_shape[normalized_dim] = 1
    else:
        del grad_output_shape[normalized_dim]

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
        lambda *args: _mean_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_mean_backward,
        (grad_output, dim, keep_dim, grad_input),
        {},
        rtol=rtol,
        atol=atol,
    )


def _mean_backward(
    grad_output,
    dim,
    keep_dim,
    grad_input,
    *,
    implementation_index,
):
    infini.ops.mean_backward(
        grad_output,
        dim,
        keep_dim,
        grad_input,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )

    return grad_input


def _torch_mean_backward(grad_output, dim, keep_dim, grad_input):
    normalized_dim = dim % grad_input.ndim
    reduce_size = grad_input.shape[normalized_dim]
    expanded_grad = grad_output

    if not keep_dim:
        expanded_grad = expanded_grad.unsqueeze(normalized_dim)

    grad_input.copy_(expanded_grad.expand_as(grad_input) / reduce_size)

    return grad_input
