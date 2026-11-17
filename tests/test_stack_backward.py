import json
import subprocess
import sys

import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


def test_stack_backward_alias(device, implementation_index):
    grad_output = torch.arange(12, dtype=torch.float32, device=device).reshape(2, 2, 3)
    expected = grad_output.clone()
    outputs = [grad_output[1], grad_output[0]]
    infini.ops.stack_backward(
        grad_output,
        0,
        outputs,
        stream=get_stream(device),
        implementation_index=implementation_index,
    )
    torch.testing.assert_close(outputs[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], expected[1], rtol=0, atol=0)


def test_stack_backward_rejects_output_overlap(device, implementation_index):
    setup = f"device = torch.device({str(device)!r})\nx = torch.ones((2, 2, 3), device=device)\n"
    probe = setup + (
        "bad = torch.empty((2, 3), dtype=torch.int32, device=device)\n"
        "infini.ops.StackBackward(x, 0, [bad, bad])\n"
    )
    source = (
        setup
        + f"infini.ops.stack_backward(x, 0, [x[0], x[0]], stream=get_stream(device), implementation_index={implementation_index})\n"
    )
    # Run the constructor probe first; isolate C++ assertions from pytest.
    prelude = (
        f"import sys; sys.path[:] = {json.dumps(sys.path)}\n"
        "import infini.ops\nimport torch\n"
        "from tests.utils import get_stream\n"
    )

    for code, expected_message in (
        (probe, None),
        (source, "Output storage must not overlap"),
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


# Format: (input_shape, num_inputs, dim, grad_output_strides, grad_input_strides)
_TEST_CASES = (
    pytest.param(
        (2, 5),
        3,
        0,
        None,
        None,
        id="two-dimensional-first-dimension",
    ),
    pytest.param(
        (2, 5),
        4,
        1,
        None,
        None,
        id="two-dimensional-second-dimension",
    ),
    pytest.param(
        (2, 3, 4),
        2,
        -1,
        None,
        None,
        id="three-dimensional-negative-dimension",
    ),
    pytest.param(
        (2, 3),
        3,
        1,
        (20, 5, 1),
        (5, 1),
        id="strided-gradients",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "input_shape",
        "num_inputs",
        "dim",
        "grad_output_strides",
        "grad_input_strides",
    ),
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
def test_stack_backward(
    input_shape,
    num_inputs,
    dim,
    grad_output_strides,
    grad_input_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    normalized_dim = dim % len(input_shape + (0,))
    grad_output_shape = (
        input_shape[:normalized_dim] + (num_inputs,) + input_shape[normalized_dim:]
    )
    grad_output = randn_strided(
        grad_output_shape,
        grad_output_strides,
        dtype=dtype,
        device=device,
    )
    grad_inputs = [
        empty_strided(
            input_shape,
            grad_input_strides,
            dtype=dtype,
            device=device,
        )
        for _ in range(num_inputs)
    ]
    for grad_input in grad_inputs:
        grad_input.fill_(17)

    return Payload(
        lambda *args: _stack_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_stack_backward,
        (grad_output, dim, grad_inputs),
        {},
        rtol=rtol,
        atol=atol,
    )


def _stack_backward(
    grad_output,
    dim,
    grad_inputs,
    *,
    implementation_index,
):
    infini.ops.stack_backward(
        grad_output,
        dim,
        grad_inputs,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )

    return torch.stack(grad_inputs, dim=dim)


def _torch_stack_backward(grad_output, dim, grad_inputs):
    for grad_input, expected in zip(grad_inputs, torch.unbind(grad_output, dim=dim)):
        grad_input.copy_(expected)

    return torch.stack(grad_inputs, dim=dim)
