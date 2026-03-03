import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, randn_strided


def _causal_softmax(out, x):
    infini.ops.causal_softmax(out, x)
    return out


def _torch_causal_softmax(out, x):
    mask = torch.tril(torch.ones_like(x), diagonal=-1).flip(dims=[-2, -1])
    masked = torch.where(mask == 1, -torch.inf, x.to(torch.float32))
    result = torch.nn.functional.softmax(masked, dim=-1, dtype=x.dtype)
    out.copy_(result)
    return out


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, x_strides, out_strides",
    (
        ((3, 3), None, None),
        ((3, 5), None, None),
        ((32, 512), None, None),
        ((32, 512), (1024, 1), (1024, 1)),
        ((4, 20, 512), None, None),
        ((4, 20, 512), (20480, 512, 1), None),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_causal_softmax(shape, x_strides, out_strides, dtype, device, rtol, atol):
    if getattr(infini.ops, "causal_softmax", None) is None:
        pytest.skip("causal_softmax not available (wrapper generation skipped)")

    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("CPU backend does not support fp16/bf16")

    x = randn_strided(shape, x_strides, dtype=dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        _causal_softmax,
        _torch_causal_softmax,
        (out, x),
        {},
        rtol=rtol,
        atol=atol,
    )
