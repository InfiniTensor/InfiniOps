import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_npu_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "a_shape, b_shape, out_shape",
    (
        ((4, 64), (64, 32), (4, 32)),
        ((2, 128), (128, 256), (2, 256)),
        ((1, 4096), (4096, 4096), (1, 4096)),
        ((2, 4, 64), (2, 64, 32), (2, 4, 32)),
        ((4, 8, 128), (4, 128, 64), (4, 8, 64)),
    ),
)
@pytest.mark.parametrize("trans_a", (False, True))
@pytest.mark.parametrize("trans_b", (False, True))
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
    a_shape,
    b_shape,
    out_shape,
    trans_a,
    trans_b,
    has_bias,
    dtype,
    device,
    rtol,
    atol,
):
    a = randn_strided(a_shape, None, dtype=dtype, device=device)
    b = randn_strided(b_shape, None, dtype=dtype, device=device)

    if trans_a:
        a = a.transpose(-2, -1)

    if trans_b:
        b = b.transpose(-2, -1)

    # Bias shape is [N], the last dim of the output.
    bias = None

    if has_bias:
        N = out_shape[-1]
        bias = randn_strided((N,), None, dtype=dtype, device=device)

    out = empty_strided(out_shape, None, dtype=dtype, device=device)

    return Payload(
        lambda *args: _linear(*args, trans_a=trans_a, trans_b=trans_b),
        lambda *args: _torch_linear(*args, trans_a=trans_a, trans_b=trans_b),
        (a, b, bias, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _linear(a, b, bias, out, trans_a=False, trans_b=False):
    if a.device.type == "npu":
        infini.ops.linear(a, b, bias, trans_a, trans_b, out, stream=get_npu_stream(a))
    else:
        infini.ops.linear(a, b, bias, trans_a, trans_b, out)

    return out


def _torch_linear(a, b, bias, out, trans_a=False, trans_b=False):
    if trans_a:
        a = a.transpose(-2, -1)

    if trans_b:
        b = b.transpose(-2, -1)

    result = torch.matmul(a.float(), b.float())

    if bias is not None:
        result = result + bias.float()

    out.copy_(result.to(out.dtype))

    return out
