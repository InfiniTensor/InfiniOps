import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, weight_shape, x1_strides, x2_strides, weight_strides, y_out_strides, x_out_strides",
    (
        ((1, 64), (64,), None, None, None, None, None),
        ((2, 128), (128,), None, None, None, None, None),
        ((4, 48, 64), (64,), None, None, None, None, None),
        ((2, 4, 2048), (2048,), None, None, None, None, None),
        ((1, 64), (64,), (64, 1), (64, 1), (1,), (64, 1), (64, 1)),
        (
            (4, 48, 64),
            (64,),
            (3072, 64, 1),
            (3072, 64, 1),
            (1,),
            (3072, 64, 1),
            (3072, 64, 1),
        ),
    ),
)
@pytest.mark.parametrize("eps", (1e-6, 1e-5))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 2e-2, 1e-2),
    ),
)
def test_add_rms_norm(
    shape,
    weight_shape,
    x1_strides,
    x2_strides,
    weight_strides,
    y_out_strides,
    x_out_strides,
    eps,
    dtype,
    device,
    rtol,
    atol,
):
    x1 = randn_strided(shape, x1_strides, dtype=dtype, device=device)
    x2 = randn_strided(shape, x2_strides, dtype=dtype, device=device)
    weight = randn_strided(weight_shape, weight_strides, dtype=dtype, device=device)
    y_out = empty_strided(shape, y_out_strides, dtype=dtype, device=device)
    x_out = empty_strided(shape, x_out_strides, dtype=dtype, device=device)

    return Payload(
        _add_rms_norm,
        _torch_add_rms_norm,
        (x1, x2, weight),
        {"eps": eps, "y_out": y_out, "x_out": x_out},
        rtol=rtol,
        atol=atol,
    )


def _add_rms_norm(x1, x2, weight, *, eps=1e-6, y_out=None, x_out=None):
    infini.ops.add_rms_norm(x1, x2, weight, eps, y_out, x_out)

    return y_out


def _torch_add_rms_norm(x1, x2, weight, *, eps=1e-6, y_out=None, x_out=None):
    # Compute residual = x1 + x2.
    residual = x1.float() + x2.float()

    if x_out is not None:
        x_out.copy_(residual.to(x1.dtype))

    # Compute rms_norm(residual) * weight.
    rms = torch.sqrt(torch.mean(residual * residual, dim=-1, keepdim=True) + eps)
    result = (residual / rms).to(x1.dtype) * weight

    if y_out is not None:
        y_out.copy_(result)
    else:
        y_out = result

    return y_out
