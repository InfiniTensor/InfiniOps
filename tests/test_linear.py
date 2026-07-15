import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, weight_shape, out_shape",
    (
        ((64,), (32, 64), (32,)),
        ((4, 64), (32, 64), (4, 32)),
        ((2, 128), (256, 128), (2, 256)),
        ((1, 4096), (4096, 4096), (1, 4096)),
        ((2, 4, 64), (32, 64), (2, 4, 32)),
        ((2, 3, 4, 64), (32, 64), (2, 3, 4, 32)),
    ),
)
@pytest.mark.parametrize("has_bias", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-2, 5e-2),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_linear(
    input_shape,
    weight_shape,
    out_shape,
    has_bias,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(input_shape, None, dtype=dtype, device=device)
    weight = randn_strided(weight_shape, None, dtype=dtype, device=device)

    # Bias shape is [N], the last dim of the output.
    bias = None

    if has_bias:
        bias = randn_strided((out_shape[-1],), None, dtype=dtype, device=device)

    out = empty_strided(out_shape, None, dtype=dtype, device=device)

    return Payload(
        _linear,
        _torch_linear,
        (input, weight, bias, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _linear(input, weight, bias, out):
    infini.ops.linear(input, weight, bias, out, stream=get_stream(input.device))

    return out


def _torch_linear(input, weight, bias, out):
    result = torch.nn.functional.linear(
        input.float(), weight.float(), None if bias is None else bias.float()
    )

    out.copy_(result.to(out.dtype))

    return out


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_linear_noncontiguous_weight(dtype, device):
    input = randn_strided((3, 8), None, dtype=dtype, device=device)
    weight = randn_strided((5, 8), (1, 5), dtype=dtype, device=device)
    out = empty_strided((3, 5), None, dtype=dtype, device=device)

    return Payload(
        _linear,
        _torch_linear,
        (input, weight, None, out),
        {},
        rtol=1e-2,
        atol=1e-2,
    )
