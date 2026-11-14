import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream, randn_strided


# Format: (shape, rate, gradient_strides, tensor_strides)
_TEST_CASES = (
    pytest.param((17,), 0.5, None, None, id="one-dimensional"),
    pytest.param((2, 3), -0.25, None, None, id="two-dimensional"),
    pytest.param(
        (2, 3, 4),
        1.5,
        (30, 7, 1),
        (35, 8, 1),
        id="strided-tensors",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, rate, gradient_strides, tensor_strides",
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
def test_accumulate_grad(
    shape,
    rate,
    gradient_strides,
    tensor_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    gradient = randn_strided(
        shape,
        gradient_strides,
        dtype=dtype,
        device=device,
    )
    tensor = randn_strided(
        shape,
        tensor_strides,
        dtype=dtype,
        device=device,
    )

    return Payload(
        lambda *args: _accumulate_grad(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_accumulate_grad,
        (gradient, rate, tensor),
        {},
        rtol=rtol,
        atol=atol,
    )


def _accumulate_grad(gradient, rate, tensor, *, implementation_index):
    infini.ops.accumulate_grad(
        gradient,
        rate,
        tensor,
        stream=get_stream(gradient.device),
        implementation_index=implementation_index,
    )

    return tensor


def _torch_accumulate_grad(gradient, rate, tensor):
    tensor.add_(gradient * rate)

    return tensor
