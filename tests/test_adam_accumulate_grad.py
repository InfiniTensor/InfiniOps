import json
import os
import subprocess
import sys

import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream, randn_strided


# Format: (shape, learning_rate, beta1, beta2, eps, step, strides)
_TEST_CASES = (
    pytest.param(
        (17,),
        1e-2,
        0.9,
        0.999,
        1e-8,
        1,
        None,
        id="one-dimensional-first-step",
    ),
    pytest.param(
        (2, 3),
        5e-3,
        0.8,
        0.99,
        1e-6,
        3,
        None,
        id="two-dimensional-third-step",
    ),
    pytest.param(
        (2, 3, 4),
        1e-2,
        0.9,
        0.999,
        1e-8,
        2,
        (30, 7, 1),
        id="strided-tensors",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, learning_rate, beta1, beta2, eps, step, strides",
    _TEST_CASES,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-6),
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 2e-2, 2e-2),
    ),
)
def test_adam_accumulate_grad(
    shape,
    learning_rate,
    beta1,
    beta2,
    eps,
    step,
    strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    grad = randn_strided(shape, strides, dtype=dtype, device=device)
    param = randn_strided(shape, strides, dtype=dtype, device=device)
    m = randn_strided(shape, strides, dtype=dtype, device=device)
    v = randn_strided(shape, strides, dtype=dtype, device=device).abs_().add_(0.1)

    return Payload(
        lambda *args: _adam_accumulate_grad(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_adam_accumulate_grad,
        (grad, learning_rate, beta1, beta2, eps, step, param, m, v),
        {},
        rtol=rtol,
        atol=atol,
    )


def _adam_accumulate_grad(
    grad,
    learning_rate,
    beta1,
    beta2,
    eps,
    step,
    param,
    m,
    v,
    *,
    implementation_index,
):
    infini.ops.adam_accumulate_grad(
        grad,
        learning_rate,
        beta1,
        beta2,
        eps,
        step,
        param,
        m,
        v,
        stream=get_stream(grad.device),
        implementation_index=implementation_index,
    )

    return torch.cat((param.reshape(-1), m.reshape(-1), v.reshape(-1)))


def _torch_adam_accumulate_grad(
    grad,
    learning_rate,
    beta1,
    beta2,
    eps,
    step,
    param,
    m,
    v,
):
    reference_param = torch.nn.Parameter(
        param.detach().to(device="cpu", dtype=torch.float64)
    )
    optimizer = torch.optim.Adam(
        [reference_param],
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        foreach=False,
    )
    optimizer.state[reference_param] = {
        "step": torch.tensor(float(step - 1)),
        "exp_avg": m.detach().to(device="cpu", dtype=torch.float64).clone(),
        "exp_avg_sq": v.detach().to(device="cpu", dtype=torch.float64).clone(),
    }
    reference_param.grad = grad.detach().to(device="cpu", dtype=torch.float64).clone()
    optimizer.step()
    param.copy_(reference_param.detach())
    m.copy_(optimizer.state[reference_param]["exp_avg"])
    v.copy_(optimizer.state[reference_param]["exp_avg_sq"])

    return torch.cat((param.reshape(-1), m.reshape(-1), v.reshape(-1)))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("gradient_scale", (0.0, 1e-7, 1.0))
def test_adam_accumulate_grad_mixed_precision_multistep(
    dtype,
    gradient_scale,
    device,
    implementation_index,
):
    param = torch.linspace(-1, 1, 12, device=device).reshape(3, 4).to(dtype)
    m = torch.zeros((3, 4), dtype=torch.float32, device=device)
    v = torch.zeros_like(m)
    expected_param, expected_m, expected_v = param.clone(), m.clone(), v.clone()

    for step in range(1, 9):
        grad = torch.full_like(param, gradient_scale * (-1 if step % 2 else 1))
        _torch_adam_accumulate_grad(
            grad, 0.01, 0.9, 0.999, 1e-8, step, expected_param, expected_m, expected_v
        )
        _adam_accumulate_grad(
            grad,
            0.01,
            0.9,
            0.999,
            1e-8,
            step,
            param,
            m,
            v,
            implementation_index=implementation_index,
        )
        torch.testing.assert_close(param, expected_param, rtol=0, atol=0)
        torch.testing.assert_close(m, expected_m, rtol=1e-5, atol=1e-12)
        torch.testing.assert_close(v, expected_v, rtol=1e-5, atol=1e-20)


@pytest.mark.parametrize(
    "dtype,gradient,moment,variance",
    (
        (torch.float32, 1e20, 0.0, 0.0),
        (torch.float32, 0.0, 1e19, 1e37),
        (torch.float64, 1e155, 0.0, 0.0),
    ),
)
def test_adam_accumulate_grad_finite_intermediates(
    dtype, gradient, moment, variance, device, implementation_index
):
    grad = torch.full((3,), gradient, dtype=dtype, device=device)
    param = torch.zeros_like(grad)
    m, v = torch.full_like(grad, moment), torch.full_like(grad, variance)
    expected = [value.clone() for value in (param, m, v)]
    _torch_adam_accumulate_grad(grad, 0.001, 0.9, 0.999, 1e-8, 1, *expected)
    _adam_accumulate_grad(
        grad,
        0.001,
        0.9,
        0.999,
        1e-8,
        1,
        param,
        m,
        v,
        implementation_index=implementation_index,
    )

    for actual, reference in zip((param, m, v), expected):
        assert torch.isfinite(actual).all().item()
        rtol = 1e-12 if dtype == torch.float64 else 1e-5
        torch.testing.assert_close(actual, reference, rtol=rtol, atol=0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("stride", (1, 2))
@pytest.mark.parametrize(
    "alias,offset",
    [("none", 0)]
    + [(name, shift) for name in ("param", "m", "v") for shift in (-1, 0, 1)],
)
def test_adam_accumulate_grad_input_output_overlap(
    dtype, stride, alias, offset, device, implementation_index
):
    backing = torch.linspace(0.25, 2.0, 128, dtype=dtype, device=device)
    positions = {"param": 8, "m": 40, "v": 72}
    param, m, v = [
        backing.as_strided((6,), (stride,), pos) for pos in positions.values()
    ]
    grad = torch.full_like(param, 0.5)

    if alias != "none":
        grad = backing.as_strided((6,), (stride,), positions[alias] + offset)

    expected = [value.clone() for value in (param, m, v)]
    _torch_adam_accumulate_grad(grad.clone(), 0.001, 0.9, 0.999, 1e-8, 1, *expected)
    expected_backing = backing.clone()

    for pos, result in zip(positions.values(), expected):
        expected_backing.as_strided((6,), (stride,), pos).copy_(result)

    _adam_accumulate_grad(
        grad,
        0.001,
        0.9,
        0.999,
        1e-8,
        1,
        param,
        m,
        v,
        implementation_index=implementation_index,
    )
    torch.testing.assert_close(backing, expected_backing, rtol=1e-5, atol=1e-8)


@pytest.mark.skipif(
    os.environ.get("INFINI_OPS_TEST_ASSERTIONS") != "1",
    reason="set INFINI_OPS_TEST_ASSERTIONS=1 with an assertions-enabled build",
)
@pytest.mark.parametrize("pair", (("param", "m"), ("param", "v"), ("m", "v")))
@pytest.mark.parametrize("layout", ("full", "partial", "strided", "interleaved"))
@pytest.mark.parametrize("reverse", (False, True))
def test_adam_accumulate_grad_rejects_output_overlap(
    pair, layout, reverse, device, implementation_index
):
    stride = 2 if layout in ("strided", "interleaved") else 1
    offsets = (8, {"full": 8, "partial": 10, "strided": 10, "interleaved": 9}[layout])

    if reverse:
        offsets = offsets[::-1]

    source = f"""
import sys
sys.path[:] = {json.dumps(sys.path)}
import infini.ops
import torch
from tests.utils import get_stream
device = torch.device({str(device)!r})
backing = torch.ones(128, device=device)
outputs = {{name: torch.ones(6, device=device) for name in ('param', 'm', 'v')}}
outputs[{pair[0]!r}] = backing.as_strided((6,), ({stride},), {offsets[0]})
outputs[{pair[1]!r}] = backing.as_strided((6,), ({stride},), {offsets[1]})
grad = torch.full((6,), 0.5, device=device)
infini.ops.adam_accumulate_grad(grad, 0.001, 0.9, 0.999, 1e-8, 1,
    outputs['param'], outputs['m'], outputs['v'],
    implementation_index={implementation_index}, stream=get_stream(device))
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )
    assert result.returncode != 0, "Overlapping Adam outputs were accepted"
    assert "output storage must not overlap" in result.stdout + result.stderr
