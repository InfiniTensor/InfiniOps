import infini.ops
import pytest
import torch

from tests.utils import Payload, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "a_shape, b_shape, out_shape",
    (
        ((1, 128), (128, 64), (1, 64)),
        ((4, 256), (256, 128), (4, 128)),
        ((2, 4, 128), (2, 128, 64), (2, 4, 64)),
    ),
)
@pytest.mark.parametrize("has_bias", (False, True))
@pytest.mark.parametrize("trans_a", (False, True))
@pytest.mark.parametrize("trans_b", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-3, 1e-3),
        (torch.float16, 5e-2, 5e-2),
        (torch.bfloat16, 5e-2, 5e-2),
    ),
)
def test_linear(
    a_shape,
    b_shape,
    out_shape,
    has_bias,
    trans_a,
    trans_b,
    dtype,
    device,
    rtol,
    atol,
):
    if device == "cpu":
        pytest.skip("CPU Linear is not implemented")

    a = randn_strided(a_shape, None, dtype=dtype, device=device)
    b = randn_strided(b_shape, None, dtype=dtype, device=device)

    if trans_a:
        a = a.transpose(-2, -1)

    if trans_b:
        b = b.transpose(-2, -1)

    out = randn_strided(out_shape, None, dtype=dtype, device=device)

    bias = None

    if has_bias:
        n = out_shape[-1]
        bias = randn_strided((n,), None, dtype=dtype, device=device)

    return Payload(
        lambda *args: _linear(*args),
        _torch_linear,
        (a, b, bias, trans_a, trans_b, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _linear(a, b, bias, trans_a, trans_b, out):
    infini.ops.linear(a, b, bias, trans_a, trans_b, out)

    return out


def _torch_linear(a, b, bias, trans_a, trans_b, out):
    a_mat = a.transpose(-2, -1) if trans_a else a
    b_mat = b.transpose(-2, -1) if trans_b else b

    try:
        result = torch.matmul(a_mat.float(), b_mat.float()).to(out.dtype)
    except RuntimeError:
        result = torch.matmul(a_mat.float(), b_mat.float()).to(out.dtype)

    if bias is not None:
        result = result + bias

    out.copy_(result)

    return out
