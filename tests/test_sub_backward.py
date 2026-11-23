import json
import subprocess
import sys

import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format: (grad_output_shape, grad_a_shape, grad_b_shape,
#          grad_output_strides, grad_a_strides, grad_b_strides)
_TEST_CASES = (
    pytest.param((17,), (17,), (17,), None, None, None, id="one-dimensional"),
    pytest.param((2, 3), (2, 3), (2, 3), None, None, None, id="two-dimensional"),
    pytest.param(
        (2, 3, 4),
        (2, 3, 4),
        (2, 3, 4),
        None,
        None,
        None,
        id="three-dimensional",
    ),
    pytest.param(
        (2, 3, 4),
        (2, 3, 4),
        (2, 3, 4),
        (30, 7, 1),
        (35, 8, 1),
        (36, 9, 1),
        id="strided-tensors",
    ),
    pytest.param(
        (2, 3),
        (2, 3),
        (2, 3),
        (1, 2),
        (1, 2),
        (1, 2),
        id="transposed-layout",
    ),
    pytest.param(
        (2, 3, 4),
        (2, 3, 4),
        (4,),
        None,
        None,
        None,
        id="broadcast-b-leading-dimensions",
    ),
    pytest.param(
        (2, 3, 4),
        (4,),
        (2, 3, 4),
        None,
        None,
        None,
        id="broadcast-a-leading-dimensions",
    ),
    pytest.param(
        (2, 3, 4),
        (2, 1, 4),
        (1, 3, 1),
        None,
        None,
        None,
        id="broadcast-both-inputs",
    ),
    pytest.param(
        (2, 3, 4),
        (2, 1, 4),
        (1, 3, 1),
        (30, 7, 1),
        (11, 7, 2),
        (15, 3, 1),
        id="broadcast-strided-tensors",
    ),
    pytest.param(
        (2, 3),
        (),
        (2, 3),
        None,
        None,
        None,
        id="broadcast-scalar-a",
    ),
    pytest.param(
        (2, 3),
        (2, 3),
        (),
        None,
        None,
        None,
        id="broadcast-scalar-b",
    ),
    pytest.param((), (), (), None, None, None, id="scalar-output"),
    pytest.param(
        (1, 3, 1),
        (3, 1),
        (1, 3, 1),
        None,
        None,
        None,
        id="singleton-dimensions",
    ),
    pytest.param(
        (0, 3),
        (0, 3),
        (0, 3),
        None,
        None,
        None,
        id="empty-output",
    ),
    pytest.param(
        (0, 3),
        (1, 3),
        (0, 3),
        None,
        None,
        None,
        id="empty-output-nonempty-grad-a",
    ),
    pytest.param(
        (0, 3),
        (0, 3),
        (1, 3),
        None,
        None,
        None,
        id="empty-output-nonempty-grad-b",
    ),
)

_ALPHAS = (
    pytest.param(1.0, id="alpha-one"),
    pytest.param(0.0, id="alpha-zero"),
    pytest.param(2.0, id="alpha-two"),
    pytest.param(-1.0, id="alpha-negative"),
    pytest.param(0.5, id="alpha-fractional"),
)

_DTYPES = (
    (torch.float32, 1e-5, 1e-6),
    (torch.float16, 1e-3, 1e-3),
    (torch.bfloat16, 1e-2, 1e-2),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "grad_output_shape, grad_a_shape, grad_b_shape, "
    "grad_output_strides, grad_a_strides, grad_b_strides",
    _TEST_CASES,
)
@pytest.mark.parametrize("alpha", _ALPHAS)
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPES)
def test_sub_backward(
    grad_output_shape,
    grad_a_shape,
    grad_b_shape,
    grad_output_strides,
    grad_a_strides,
    grad_b_strides,
    alpha,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    grad_output = randn_strided(
        grad_output_shape,
        grad_output_strides,
        dtype=dtype,
        device=device,
    )
    grad_a = empty_strided(
        grad_a_shape,
        grad_a_strides,
        dtype=dtype,
        device=device,
    )
    grad_b = empty_strided(
        grad_b_shape,
        grad_b_strides,
        dtype=dtype,
        device=device,
    )
    grad_a.fill_(17)
    grad_b.fill_(17)

    return Payload(
        lambda *args: _sub_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_sub_backward,
        (grad_output, alpha, grad_a, grad_b),
        {},
        rtol=rtol,
        atol=atol,
    )


def _pack_result(grad_output, grad_a, grad_b):
    # Also compare grad_output: ordinary, non-alias calls must not modify it.
    return torch.cat(
        (grad_a.reshape(-1), grad_b.reshape(-1), grad_output.reshape(-1)),
    )


def _sub_backward(grad_output, alpha, grad_a, grad_b, *, implementation_index):
    infini.ops.sub_backward(
        grad_output,
        alpha,
        grad_a,
        grad_b,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )
    return _pack_result(grad_output, grad_a, grad_b)


def _torch_sub_backward(grad_output, alpha, grad_a, grad_b):
    # Use canonical forward + autograd, independently of the hand-written formula.
    with torch.enable_grad():
        a = torch.zeros_like(grad_a, requires_grad=True)
        b = torch.zeros_like(grad_b, requires_grad=True)
        out = torch.sub(a, b, alpha=alpha)
        expected_a, expected_b = torch.autograd.grad(
            out,
            (a, b),
            grad_outputs=grad_output,
        )

    # Autograd may return a view of grad_output. Snapshot BOTH results first.
    expected_a = expected_a.clone()
    expected_b = expected_b.clone()
    with torch.no_grad():
        grad_a.copy_(expected_a)
        grad_b.copy_(expected_b)
    return _pack_result(grad_output, grad_a, grad_b)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("target", ("grad_a", "grad_b"))
@pytest.mark.parametrize("layout", ("exact", "partial", "strided", "broadcast"))
@pytest.mark.parametrize("alpha", _ALPHAS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_sub_backward_alias(
    target,
    layout,
    alpha,
    implementation_index,
    dtype,
    device,
):
    # Small integers and power-of-two alpha make these cases exactly representable.
    source = torch.arange(-16, 16, dtype=dtype, device=device)

    return Payload(
        lambda values: _run_alias(
            values,
            alpha,
            target,
            layout,
            implementation_index=implementation_index,
        ),
        lambda values: _run_alias(values, alpha, target, layout),
        (source,),
        {},
        rtol=0,
        atol=0,
    )


def _run_alias(source, alpha, target, layout, *, implementation_index=None):
    # Construct views INSIDE each invocation so Payload copying cannot remove aliasing.
    storage = source.clone()
    grad_output = storage[:6].view(2, 3)
    if layout == "exact":
        shared_output = grad_output
    elif layout == "partial":
        shared_output = storage[1:7].view(2, 3)
    elif layout == "strided":
        shared_output = storage[:6].view(3, 2).transpose(0, 1)
    elif layout == "broadcast":
        shared_output = storage[:3].view(1, 3)
    else:
        raise ValueError(f"Unknown alias layout: {layout}")

    independent_output = torch.full_like(grad_output, 17)
    if target == "grad_a":
        grad_a, grad_b = shared_output, independent_output
    else:
        grad_a, grad_b = independent_output, shared_output

    if implementation_index is None:
        _torch_sub_backward(grad_output, alpha, grad_a, grad_b)
    else:
        _sub_backward(
            grad_output,
            alpha,
            grad_a,
            grad_b,
            implementation_index=implementation_index,
        )

    # Include the whole backing buffer to detect writes outside the output view.
    return torch.cat((grad_a.reshape(-1), grad_b.reshape(-1), storage))


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "alpha, grad_value, grad_a_shape, grad_b_shape, expected_a, expected_b",
    (
        pytest.param(2.0, 1.0, (2, 3), (2, 3), 1.0, -2.0, id="review-example"),
        pytest.param(2.0, 1.0, (2, 3), (3,), 1.0, -4.0, id="scaled-reduction"),
        pytest.param(0.0, 1.0, (2, 3), (3,), 1.0, 0.0, id="zero-alpha"),
        pytest.param(-1.0, 1.0, (2, 3), (3,), 1.0, 2.0, id="negative-alpha"),
        pytest.param(0.5, 1.0, (2, 3), (3,), 1.0, -1.0, id="fractional-alpha"),
        pytest.param(2.0, 1.0, (1, 3), (2, 1), 2.0, -6.0, id="both-reduced"),
        pytest.param(
            0.5, 40000.0, (2, 3), (2, 3), 40000.0, -20000.0, id="large-finite-gradient"
        ),
        pytest.param(
            0.0, 40000.0, (2, 3), (3,), 40000.0, 0.0, id="zero-alpha-before-reduction"
        ),
    ),
)
def test_sub_backward_known_values(
    alpha,
    grad_value,
    grad_a_shape,
    grad_b_shape,
    expected_a,
    expected_b,
    implementation_index,
    dtype,
    device,
):
    grad_output = torch.full((2, 3), grad_value, dtype=dtype, device=device)
    grad_a = torch.full(grad_a_shape, 17.0, dtype=dtype, device=device)
    grad_b = torch.full(grad_b_shape, 17.0, dtype=dtype, device=device)
    expected = _pack_result(
        grad_output,
        torch.full_like(grad_a, expected_a),
        torch.full_like(grad_b, expected_b),
    )

    return Payload(
        lambda *args: _sub_backward(
            *args,
            implementation_index=implementation_index,
        ),
        lambda *args: expected.clone(),
        (grad_output, alpha, grad_a, grad_b),
        {},
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("offset", (0, 2))
@pytest.mark.parametrize("strides", ((3, 1), (6, 2)))
def test_sub_backward_rejects_output_overlap(
    offset, strides, device, implementation_index
):
    setup = f"device = torch.device({str(device)!r})\nx = torch.ones((2, 3), device=device)\n"
    probe = setup + "infini.ops.SubBackward(x, 2.0, torch.empty(4, device=device), x)\n"
    source = (
        setup
        + f"""
backing = torch.ones(64, device=device)
a = backing.as_strided((2, 3), {strides!r}, 0)
b = backing.as_strided((2, 3), {strides!r}, {offset})
infini.ops.sub_backward(x, 2.0, a, b,
    stream=get_stream(device), implementation_index={implementation_index})
"""
    )
    # Run the constructor probe first; isolate C++ assertions from pytest.
    prelude = (
        f"import sys; sys.path[:] = {json.dumps(sys.path)}\n"
        "import infini.ops\nimport torch\n"
        "from tests.utils import get_stream\n"
    )

    for code, expected_message in (
        (probe, None),
        (source, "output storage must not overlap"),
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
