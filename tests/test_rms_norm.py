import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, randn_strided


def _rms_norm(x, w, out, *, epsilon=1e-6):
    infini.ops.rms_norm(out, x, w, epsilon)

    return out


def _torch_rms_norm(x, w, out, *, epsilon=1e-6):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + epsilon)
    result = x * w / rms
    out.copy_(result)

    return out


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "x_shape, w_shape, x_strides, w_strides, out_strides",
    (
        ((1, 64), (64,), None, None, None),
        ((2, 128), (128,), None, None, None),
        ((4, 48, 64), (64,), None, None, None),
        ((2, 4, 2048), (2048,), None, None, None),
        ((1, 64), (64,), (64, 1), (1,), (64, 1)),
        ((4, 48, 64), (64,), (3072, 64, 1), (1,), (3072, 64, 1)),
    ),
)
@pytest.mark.parametrize("epsilon", (1e-6, 1e-5))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_rms_norm(
    x_shape, w_shape, x_strides, w_strides, out_strides, epsilon, dtype, device, rtol, atol
):
    if getattr(infini.ops, "rms_norm", None) is None:
        pytest.skip("rms_norm not available (wrapper generation skipped)")

    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("CPU backend does not support fp16/bf16")

    x = randn_strided(x_shape, x_strides, dtype=dtype, device=device)
    w = randn_strided(w_shape, w_strides, dtype=dtype, device=device)
    out = empty_strided(x_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        _rms_norm, _torch_rms_norm, (x, w, out), {"epsilon": epsilon}, rtol=rtol, atol=atol
    )
