import infini.ops
import pytest
import torch

from tests.utils import empty_strided, get_available_devices


def _torch_rms_norm(x, w, epsilon=1e-6):
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + epsilon)
    return x * w / rms


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
@pytest.mark.parametrize("epsilon", (1e-6, 1e-5))
@pytest.mark.parametrize(
    "x_shape, w_shape",
    (
        ((1, 64), (64,)),
        ((2, 128), (128,)),
        ((4, 48, 64), (64,)),
        ((2, 4, 2048), (2048,)),
    ),
)
def test_rms_norm(x_shape, w_shape, epsilon, dtype, device, rtol, atol):
    rms_norm = getattr(infini.ops, "rms_norm", None)
    if rms_norm is None:
        pytest.skip("rms_norm not available (wrapper generation skipped)")

    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("CPU backend does not support fp16/bf16")

    x = empty_strided(x_shape, None, dtype=dtype, device=device)
    w = empty_strided(w_shape, None, dtype=dtype, device=device)
    output = torch.empty_like(x)

    x.normal_()
    w.uniform_(0.5, 1.5)

    rms_norm(output, x, w, epsilon)
    expected = _torch_rms_norm(x, w, epsilon)

    assert torch.allclose(output, expected, rtol=rtol, atol=atol)
