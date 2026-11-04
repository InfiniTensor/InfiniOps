import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format: (shape, grad_output_strides, grad_a_strides, grad_b_strides)
_TEST_CASES = (
    pytest.param((17,), None, None, None, id="one-dimensional"),
    pytest.param((2, 3), None, None, None, id="two-dimensional"),
    pytest.param((2, 3, 4), None, None, None, id="three-dimensional"),
    pytest.param(
        (2, 3, 4),
        (30, 7, 1),
        (35, 8, 1),
        (36, 9, 1),
        id="strided-tensors",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, grad_output_strides, grad_a_strides, grad_b_strides",
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
def test_sub_backward(
    shape,
    grad_output_strides,
    grad_a_strides,
    grad_b_strides,
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
    grad_a = empty_strided(shape, grad_a_strides, dtype=dtype, device=device)
    grad_b = empty_strided(shape, grad_b_strides, dtype=dtype, device=device)
    grad_a.fill_(17)
    grad_b.fill_(17)

    return Payload(
        lambda *args: _sub_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_sub_backward,
        (grad_output, grad_a, grad_b),
        {},
        rtol=rtol,
        atol=atol,
    )


def _sub_backward(grad_output, grad_a, grad_b, *, implementation_index):
    infini.ops.sub_backward(
        grad_output,
        grad_a,
        grad_b,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )

    return torch.cat((grad_a.reshape(-1), grad_b.reshape(-1)))


def _torch_sub_backward(grad_output, grad_a, grad_b):
    grad_a.copy_(grad_output)
    grad_b.copy_(-grad_output)

    return torch.cat((grad_a.reshape(-1), grad_b.reshape(-1)))
