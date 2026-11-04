import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_TEST_CASES = (
    pytest.param(
        (12,),
        None,
        3,
        0,
        id="one-dimensional-even",
    ),
    pytest.param(
        (10,),
        None,
        4,
        0,
        id="one-dimensional-uneven",
    ),
    pytest.param(
        (4, 12),
        None,
        3,
        1,
        id="two-dimensional",
    ),
    pytest.param(
        (2, 5, 8),
        None,
        3,
        -1,
        id="negative-dimension",
    ),
    pytest.param(
        (4, 6),
        (12, 2),
        2,
        1,
        id="strided-input",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, input_strides, split_size, dim",
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
def test_split(
    input_shape,
    input_strides,
    split_size,
    dim,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(input_shape, input_strides, dtype=dtype, device=device)
    output_shapes = tuple(
        output.shape for output in torch.split(input, split_size, dim=dim)
    )
    outs = [
        empty_strided(shape, None, dtype=dtype, device=device)
        for shape in output_shapes
    ]

    return Payload(
        lambda *args: _split(
            *args,
            split_size=split_size,
            dim=dim,
            implementation_index=implementation_index,
        ),
        lambda *args: _torch_split(
            *args,
            split_size=split_size,
            dim=dim,
        ),
        (input, outs),
        {},
        rtol=rtol,
        atol=atol,
    )


def _split(input, outs, *, split_size, dim, implementation_index):
    infini.ops.split(
        input,
        split_size,
        dim,
        outs,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return torch.cat(outs, dim=dim)


def _torch_split(input, outs, *, split_size, dim):
    for out, expected in zip(outs, torch.split(input, split_size, dim=dim)):
        out.copy_(expected)

    return torch.cat(outs, dim=dim)
