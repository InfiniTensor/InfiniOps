import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, rand_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "out_shape, input_strides, out_strides",
    (
        ((13, 4), None, None),
        ((13, 4), (12, 1), (6, 1)),
        ((13, 4, 4), None, None),
        ((13, 4, 4), (48, 10, 1), (24, 6, 1)),
        ((16, 5632), None, None),
        ((16, 5632), (13312, 1), (6656, 1)),
        ((4, 4, 5632), None, None),
        ((4, 4, 5632), (53248, 13312, 1), (26624, 6656, 1)),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_silu_and_mul(
    out_shape,
    input_strides,
    out_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input_shape = (*out_shape[:-1], out_shape[-1] * 2)
    input = rand_strided(input_shape, input_strides, dtype=dtype, device=device)
    out = empty_strided(out_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _silu_and_mul(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _torch_silu_and_mul,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _silu_and_mul(input, out, implementation_index=0):
    infini.ops.silu_and_mul(
        input,
        out,
        implementation_index=implementation_index,
        stream=get_stream(input.device),
    )

    return out


def _torch_silu_and_mul(input, out):
    hidden_size = input.shape[-1] // 2
    gate = input[..., :hidden_size]
    up = input[..., hidden_size:]

    return torch.mul(torch.nn.functional.silu(gate), up, out=out)
