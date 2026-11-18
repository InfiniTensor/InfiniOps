import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format: (input_shape, repeats, dim, input_strides, out_strides)
_TEST_CASES = (
    pytest.param(
        (5,),
        3,
        0,
        None,
        None,
        id="one-dimensional",
    ),
    pytest.param(
        (2, 3),
        2,
        0,
        None,
        None,
        id="two-dimensional-first-dimension",
    ),
    pytest.param(
        (2, 3),
        4,
        1,
        None,
        None,
        id="two-dimensional-second-dimension",
    ),
    pytest.param(
        (2, 3, 4),
        2,
        -1,
        None,
        None,
        id="three-dimensional-negative-dimension",
    ),
    pytest.param(
        (2, 3, 4),
        1,
        1,
        None,
        None,
        id="repeat-once",
    ),
    pytest.param(
        (2, 3, 4),
        2,
        1,
        (20, 6, 1),
        (32, 5, 1),
        id="strided-input-and-output",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, repeats, dim, input_strides, out_strides",
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
def test_repeat_interleave(
    input_shape,
    repeats,
    dim,
    input_strides,
    out_strides,
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
    out_shape = list(input_shape)
    out_shape[dim % len(input_shape)] *= repeats
    out = empty_strided(
        tuple(out_shape),
        out_strides,
        dtype=dtype,
        device=device,
    )
    out.fill_(17)

    return Payload(
        lambda *args: _repeat_interleave(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_repeat_interleave,
        (input, repeats, dim, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _repeat_interleave(input, repeats, dim, out, *, implementation_index):
    infini.ops.repeat_interleave(
        input,
        repeats,
        dim,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_repeat_interleave(input, repeats, dim, out):
    out.copy_(torch.repeat_interleave(input, repeats, dim=dim))

    return out
