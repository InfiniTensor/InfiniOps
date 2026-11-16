import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, weight_shape, weight_strides, out_shape",
    (
        ((64,), (32, 64), None, (32,)),
        ((4, 64), (32, 64), None, (4, 32)),
        ((2, 128), (256, 128), None, (2, 256)),
        ((1, 4096), (4096, 4096), None, (1, 4096)),
        ((2, 4, 64), (32, 64), None, (2, 4, 32)),
        ((2, 3, 4, 64), (32, 64), None, (2, 3, 4, 32)),
        ((3, 8), (5, 8), (1, 5), (3, 5)),
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
    weight_strides,
    out_shape,
    has_bias,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(input_shape, None, dtype=dtype, device=device)
    weight = randn_strided(weight_shape, weight_strides, dtype=dtype, device=device)

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


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, input_strides, weight_shape, weight_strides, out_shape, out_strides",
    (
        ((4, 64), (96, 1), (32, 64), None, (4, 32), (64, 1)),
        ((8,), (2,), (4, 8), (1, 4), (4,), (2,)),
        ((2, 4, 8), (96, 24, 2), (5, 8), None, (2, 4, 5), (30, 6, 1)),
        ((3, 16), (32, 1), (8, 16), None, (3, 8), (16, 1)),
        ((2, 3, 16), (96, 32, 1), (8, 16), (1, 8), (2, 3, 8), None),
    ),
)
@pytest.mark.parametrize("bias_strides", (None, (2,)))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-2, 5e-2),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_linear_strided_layouts(
    input_shape,
    input_strides,
    weight_shape,
    weight_strides,
    out_shape,
    out_strides,
    bias_strides,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(input_shape, input_strides, dtype=dtype, device=device)
    weight = randn_strided(weight_shape, weight_strides, dtype=dtype, device=device)
    bias = randn_strided((out_shape[-1],), bias_strides, dtype=dtype, device=device)
    out = empty_strided(out_shape, out_strides, dtype=dtype, device=device)

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
    if input.device.type == "musa":
        with torch.backends.mudnn.flags(allow_tf32=False):
            result = torch.nn.functional.linear(
                input.float().contiguous(),
                weight.float().contiguous(),
                None if bias is None else bias.float().contiguous(),
            )

            out.copy_(result.to(out.dtype))

            return out

    result = torch.nn.functional.linear(
        input.float(), weight.float(), None if bias is None else bias.float()
    )

    out.copy_(result.to(out.dtype))

    return out
