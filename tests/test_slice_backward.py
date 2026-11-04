import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_TEST_CASES = (
    pytest.param(
        (17,),
        (5,),
        None,
        None,
        (1,),
        (16,),
        (3,),
        id="one-dimensional",
    ),
    pytest.param(
        (6, 9),
        (3, 3),
        (12, 1),
        (5, 1),
        (1, 0),
        (6, 9),
        (2, 3),
        id="strided-two-dimensional",
    ),
    pytest.param(
        (4, 6, 7),
        (2, 3, 3),
        (60, 9, 1),
        (20, 5, 1),
        (-4, 1, 0),
        (-1, 6, 7),
        (2, 2, 3),
        id="negative-indices",
    ),
    pytest.param(
        (2, 3, 4, 5),
        (2, 2, 3, 3),
        (80, 25, 6, 1),
        None,
        (0, 0, 1, 0),
        (2, 3, 4, 5),
        (1, 2, 1, 2),
        id="four-dimensional",
    ),
    pytest.param(
        (5, 7),
        (0, 7),
        None,
        None,
        (3, 0),
        (3, 7),
        (1, 1),
        id="empty-slice",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "grad_input_shape",
        "grad_output_shape",
        "grad_input_strides",
        "grad_output_strides",
        "starts",
        "ends",
        "steps",
    ),
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
def test_slice_backward(
    grad_input_shape,
    grad_output_shape,
    grad_input_strides,
    grad_output_strides,
    starts,
    ends,
    steps,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    grad_output = randn_strided(
        grad_output_shape,
        grad_output_strides,
        dtype=dtype,
        device=device,
    )
    grad_input = empty_strided(
        grad_input_shape,
        grad_input_strides,
        dtype=dtype,
        device=device,
    )
    grad_input.fill_(17)

    return Payload(
        lambda *args: _slice_backward(*args, implementation_index=implementation_index),
        _torch_slice_backward,
        (grad_output, starts, ends, steps, grad_input),
        {},
        rtol=rtol,
        atol=atol,
    )


def _slice_backward(
    grad_output,
    starts,
    ends,
    steps,
    grad_input,
    *,
    implementation_index,
):
    infini.ops.slice_backward(
        grad_output,
        starts,
        ends,
        steps,
        grad_input,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )

    return grad_input


def _torch_slice_backward(grad_output, starts, ends, steps, grad_input):
    grad_input.zero_()
    grad_input[_slice_index(starts, ends, steps)] = grad_output

    return grad_input


def _slice_index(starts, ends, steps):
    assert len(starts) == len(ends) == len(steps)

    return tuple(
        slice(start, end, step) for start, end, step in zip(starts, ends, steps)
    )
