import json
import subprocess
import sys

import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream, randn_strided


# Format: (shape, rate, gradient_strides, tensor_strides)
_TEST_CASES = (
    pytest.param((17,), 0.5, None, None, id="one-dimensional"),
    pytest.param((2, 3), -0.25, None, None, id="two-dimensional"),
    pytest.param(
        (2, 3, 4),
        1.5,
        (30, 7, 1),
        (35, 8, 1),
        id="strided-tensors",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, rate, gradient_strides, tensor_strides",
    _TEST_CASES,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_accumulate_grad(
    shape,
    rate,
    gradient_strides,
    tensor_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    gradient = randn_strided(
        shape,
        gradient_strides,
        dtype=dtype,
        device=device,
    )
    tensor = randn_strided(
        shape,
        tensor_strides,
        dtype=dtype,
        device=device,
    )

    return Payload(
        lambda *args: _accumulate_grad(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_accumulate_grad,
        (gradient, rate, tensor),
        {},
        rtol=rtol,
        atol=atol,
    )


def _accumulate_grad(gradient, rate, tensor, *, implementation_index):
    infini.ops.accumulate_grad(
        gradient,
        rate,
        tensor,
        stream=get_stream(gradient.device),
        implementation_index=implementation_index,
    )

    return tensor


def _torch_accumulate_grad(gradient, rate, tensor):
    tensor.add_(gradient * rate)

    return tensor


@pytest.mark.parametrize("rate", (0.0, 0.5, -1.0))
def test_accumulate_grad_repeated(rate, device, implementation_index):
    tensor = torch.ones(4, device=device)
    gradient = torch.full_like(tensor, 2)
    expected = tensor.clone()

    for _ in range(5):
        expected.add_(gradient, alpha=rate)
        _accumulate_grad(
            gradient, rate, tensor, implementation_index=implementation_index
        )
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)
@pytest.mark.parametrize("rate", (0.0, 0.5))
@pytest.mark.parametrize("stride", (1, 2))
@pytest.mark.parametrize("offset", (-1, 0, 1, 80))
def test_accumulate_grad_external_overlap(
    dtype, rate, stride, offset, device, implementation_index
):
    backing = torch.arange(256, dtype=dtype, device=device)
    tensor = backing.as_strided((32,), (stride,), 8)
    gradient = backing.as_strided((32,), (stride,), 8 + offset)
    expected = tensor.clone().add_(gradient.clone(), alpha=rate)
    expected_backing = backing.clone()
    expected_backing.as_strided((32,), (stride,), 8).copy_(expected)
    _accumulate_grad(gradient, rate, tensor, implementation_index=implementation_index)
    torch.testing.assert_close(backing, expected_backing, rtol=0, atol=0)


def test_accumulate_grad_rejects_integer(device, implementation_index):
    setup = f"x = torch.ones(4, dtype=torch.int32, device={str(device)!r})\n"
    probe = setup + "infini.ops.AccumulateGrad(x, 0.5, x)\n"
    source = (
        setup
        + f"infini.ops.accumulate_grad(x, 0.5, x, implementation_index={implementation_index}, stream=get_stream(x.device))\n"
    )
    # Run the constructor probe first; isolate C++ assertions from pytest.
    prelude = (
        f"import sys; sys.path[:] = {json.dumps(sys.path)}\n"
        "import infini.ops\nimport torch\n"
        "from tests.utils import get_stream\n"
    )

    for code, expected_message in (
        (probe, None),
        (source, "requires floating-point tensors"),
    ):
        result = subprocess.run(
            [sys.executable, "-c", prelude + code],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60,
        )
        diagnostic = result.stdout + result.stderr

        if expected_message is None:
            if result.returncode == 0:
                pytest.skip(
                    "descriptor validation requires an assertions-enabled build"
                )

            assert "assert" in diagnostic.lower(), diagnostic
        else:
            assert result.returncode != 0, "Invalid parameters were accepted"
            assert expected_message in diagnostic, diagnostic
