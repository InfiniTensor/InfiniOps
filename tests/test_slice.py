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
        None,
        (24, 12, 4, 1),
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
        "input_shape",
        "out_shape",
        "input_strides",
        "out_strides",
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
def test_slice(
    input_shape,
    out_shape,
    input_strides,
    out_strides,
    starts,
    ends,
    steps,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(
        input_shape, input_strides, dtype=dtype, device=device
    )
    out = empty_strided(out_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args: _slice(
            *args, implementation_index=implementation_index
        ),
        _torch_slice,
        (input, starts, ends, steps, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _slice(input, starts, ends, steps, out, *, implementation_index):
    infini.ops.slice(
        input,
        starts,
        ends,
        steps,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_slice(input, starts, ends, steps, out):
    out.copy_(input[_slice_index(starts, ends, steps)])

    return out


def _slice_index(starts, ends, steps):
    return tuple(
        slice(start, end, step)
        for start, end, step in zip(starts, ends, steps, strict=True)
    )
