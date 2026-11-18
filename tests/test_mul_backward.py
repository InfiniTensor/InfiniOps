import json
import subprocess
import sys

import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_DTYPES = (
    (torch.float32, 1e-6, 1e-6),
    (torch.float16, 1e-3, 1e-3),
    (torch.bfloat16, 1e-2, 1e-2),
)

# Format: (shape, grad_output_strides, a_strides, b_strides,
#          grad_a_strides, grad_b_strides)
_TEST_CASES = (
    pytest.param((17,), None, None, None, None, None, id="one-dimensional"),
    pytest.param((2, 3), None, None, None, None, None, id="two-dimensional"),
    pytest.param((2, 3, 4), None, None, None, None, None, id="three-dimensional"),
    pytest.param(
        (2, 3, 4),
        (30, 7, 1),
        (28, 6, 1),
        (32, 5, 1),
        (35, 8, 1),
        (36, 9, 1),
        id="strided-tensors",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "shape",
        "grad_output_strides",
        "a_strides",
        "b_strides",
        "grad_a_strides",
        "grad_b_strides",
    ),
    _TEST_CASES,
)
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPES)
def test_mul_backward(
    shape,
    grad_output_strides,
    a_strides,
    b_strides,
    grad_a_strides,
    grad_b_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    grad_output = randn_strided(
        shape,
        grad_output_strides,
        dtype=dtype,
        device=device,
    )
    a = randn_strided(shape, a_strides, dtype=dtype, device=device)
    b = randn_strided(shape, b_strides, dtype=dtype, device=device)
    grad_a = empty_strided(shape, grad_a_strides, dtype=dtype, device=device)
    grad_b = empty_strided(shape, grad_b_strides, dtype=dtype, device=device)
    grad_a.fill_(17)
    grad_b.fill_(17)

    return Payload(
        lambda *args: _mul_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_mul_backward,
        (grad_output, a, b, grad_a, grad_b),
        {},
        rtol=rtol,
        atol=atol,
    )


def _mul_backward(
    grad_output,
    a,
    b,
    grad_a,
    grad_b,
    *,
    implementation_index,
):
    infini.ops.mul_backward(
        grad_output,
        a,
        b,
        grad_a,
        grad_b,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )
    return torch.cat((grad_a.reshape(-1), grad_b.reshape(-1)))


def _torch_mul_backward(grad_output, a, b, grad_a, grad_b):
    # Finish both computations before writing potentially aliased outputs.
    result_a = (grad_output * b).sum_to_size(a.shape)
    result_b = (grad_output * a).sum_to_size(b.shape)
    grad_a.copy_(result_a)
    grad_b.copy_(result_b)
    return torch.cat((grad_a.reshape(-1), grad_b.reshape(-1)))


# Alias tests deliberately call the operator directly. A harness that clones
# each argument independently may destroy alias relationships.
def _reference_gradients(grad_output, a, b):
    # Snapshot BEFORE calling the operator: it may overwrite these inputs.
    with torch.enable_grad():
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)
        return torch.autograd.grad(
            a_ref * b_ref,
            (a_ref, b_ref),
            grad_output.detach().clone(),
        )


def _assert_gradients(grad_a, grad_b, expected, rtol, atol):
    torch.testing.assert_close(grad_a, expected[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(grad_b, expected[1], rtol=rtol, atol=atol)


def _make_inputs(shape, dtype, device):
    return {
        name: randn_strided(shape, None, dtype=dtype, device=device)
        for name in ("grad_output", "a", "b")
    }


def _backing(dtype, device):
    # Small, exactly representable values, with space for guard elements.
    return (torch.arange(40, dtype=torch.float32, device=device) / 8 - 2).to(
        dtype=dtype,
    )


def _alias_views(backing, layout):
    # All views have shape (2, 3) and internally non-overlapping elements.
    # Offset 2 leaves leading guard elements in the backing allocation.
    if layout == "full":
        x = backing.as_strided((2, 3), (3, 1), 2)
        y = backing.as_strided((2, 3), (3, 1), 2)
    elif layout == "partial":
        x = backing.as_strided((2, 3), (3, 1), 2)
        y = backing.as_strided((2, 3), (3, 1), 4)
    elif layout == "transposed":
        x = backing.as_strided((2, 3), (1, 2), 2)
        y = backing.as_strided((2, 3), (3, 1), 2)
    elif layout == "strided-partial":
        x = backing.as_strided((2, 3), (6, 2), 2)
        y = backing.as_strided((2, 3), (6, 2), 4)
    else:
        raise AssertionError(layout)
    return x, y


@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPES)
def test_mul_backward_review_regression(
    dtype,
    rtol,
    atol,
    device,
    implementation_index,
):
    a = torch.tensor([2, 3], dtype=dtype, device=device)
    b = torch.tensor([5, 7], dtype=dtype, device=device)
    grad_output = torch.ones_like(a)
    grad_a = a.view_as(a)
    grad_b = torch.full_like(b, 17)

    _mul_backward(
        grad_output,
        a,
        b,
        grad_a,
        grad_b,
        implementation_index=implementation_index,
    )

    torch.testing.assert_close(
        grad_a,
        torch.tensor([5, 7], dtype=dtype, device=device),
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        grad_b,
        torch.tensor([2, 3], dtype=dtype, device=device),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPES)
@pytest.mark.parametrize("source", ("a", "b", "grad_output"))
@pytest.mark.parametrize("destination", ("grad_a", "grad_b"))
@pytest.mark.parametrize(
    "layout",
    ("full", "partial", "transposed", "strided-partial"),
)
@pytest.mark.parametrize("reverse", (False, True), ids=("forward", "reverse"))
def test_mul_backward_input_output_alias(
    source,
    destination,
    layout,
    reverse,
    dtype,
    rtol,
    atol,
    device,
    implementation_index,
):
    inputs = _make_inputs((2, 3), dtype, device)
    backing = _backing(dtype, device)
    input_view, output_view = _alias_views(backing, layout)
    if reverse:
        input_view, output_view = output_view, input_view
    inputs[source] = input_view
    outputs = {
        name: torch.full((2, 3), 17, dtype=dtype, device=device)
        for name in ("grad_a", "grad_b")
    }
    outputs[destination] = output_view

    expected = _reference_gradients(**inputs)
    expected_backing = backing.clone()
    expected_view = expected_backing.as_strided(
        output_view.shape,
        output_view.stride(),
        output_view.storage_offset(),
    )
    expected_view.copy_(expected[0 if destination == "grad_a" else 1])
    unchanged_inputs = {
        name: tensor.clone() for name, tensor in inputs.items() if name != source
    }

    _mul_backward(**inputs, **outputs, implementation_index=implementation_index)

    _assert_gradients(outputs["grad_a"], outputs["grad_b"], expected, rtol, atol)
    # Include gaps and guard elements, not just the logical output elements.
    torch.testing.assert_close(backing, expected_backing, rtol=rtol, atol=atol)
    for name, snapshot in unchanged_inputs.items():
        torch.testing.assert_close(inputs[name], snapshot, rtol=0, atol=0)


@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPES)
@pytest.mark.parametrize(
    ("source_a", "source_b"),
    (
        ("a", "b"),
        ("b", "a"),
        ("grad_output", "a"),
        ("grad_output", "b"),
        ("a", "grad_output"),
        ("b", "grad_output"),
    ),
)
def test_mul_backward_both_outputs_alias_inputs(
    source_a,
    source_b,
    dtype,
    rtol,
    atol,
    device,
    implementation_index,
):
    inputs = _make_inputs((2, 3), dtype, device)
    grad_a = inputs[source_a].view_as(inputs[source_a])
    grad_b = inputs[source_b].view_as(inputs[source_b])
    expected = _reference_gradients(**inputs)

    _mul_backward(
        **inputs,
        grad_a=grad_a,
        grad_b=grad_b,
        implementation_index=implementation_index,
    )

    _assert_gradients(grad_a, grad_b, expected, rtol, atol)


@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPES)
@pytest.mark.parametrize("destination", ("separate", "grad_a", "grad_b"))
def test_mul_backward_all_inputs_alias(
    destination,
    dtype,
    rtol,
    atol,
    device,
    implementation_index,
):
    shared = _backing(dtype, device).as_strided((2, 3), (6, 2), 2)
    inputs = {
        name: shared.as_strided(shared.shape, shared.stride())
        for name in ("grad_output", "a", "b")
    }
    outputs = {
        name: torch.full((2, 3), 17, dtype=dtype, device=device)
        for name in ("grad_a", "grad_b")
    }
    if destination != "separate":
        outputs[destination] = shared
    expected = _reference_gradients(**inputs)

    _mul_backward(**inputs, **outputs, implementation_index=implementation_index)

    _assert_gradients(outputs["grad_a"], outputs["grad_b"], expected, rtol, atol)


@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPES)
@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    (((2, 3, 4), (4,)), ((2, 1, 4), (1, 3, 1))),
)
@pytest.mark.parametrize("alias", ("none", "a", "b", "both", "grad-to-a", "grad-to-b"))
def test_mul_backward_broadcast_alias(
    a_shape,
    b_shape,
    alias,
    dtype,
    rtol,
    atol,
    device,
    implementation_index,
):
    a = randn_strided(a_shape, None, dtype=dtype, device=device)
    b = randn_strided(b_shape, None, dtype=dtype, device=device)
    grad_output = randn_strided((2, 3, 4), None, dtype=dtype, device=device)
    grad_a = torch.full(a_shape, 17, dtype=dtype, device=device)
    grad_b = torch.full(b_shape, 17, dtype=dtype, device=device)
    if alias in ("a", "both"):
        grad_a = a.view_as(a)
    if alias in ("b", "both"):
        grad_b = b.view_as(b)
    if alias == "grad-to-a":
        grad_a = (
            grad_output.view_as(grad_output)
            if a_shape == (2, 3, 4)
            else grad_output[:, :1, :]
        )
    if alias == "grad-to-b":
        grad_b = grad_output[0, 0, :] if b_shape == (4,) else grad_output[:1, :, :1]
    expected = _reference_gradients(grad_output, a, b)

    _mul_backward(
        grad_output,
        a,
        b,
        grad_a,
        grad_b,
        implementation_index=implementation_index,
    )

    _assert_gradients(grad_a, grad_b, expected, rtol, atol)


@pytest.mark.parametrize("layout", ("full", "partial", "strided", "interleaved"))
@pytest.mark.parametrize("reverse", (False, True))
def test_mul_backward_reject_output_overlap(
    layout, reverse, device, implementation_index
):
    setup = f"device = torch.device({str(device)!r})\nx = torch.ones((2, 3), device=device)\n"
    probe = (
        setup + "infini.ops.MulBackward(x, x, x, torch.empty(4, device=device), x)\n"
    )
    stride = (6, 2) if layout in ("strided", "interleaved") else (3, 1)
    offset = {"full": 2, "partial": 4, "strided": 4, "interleaved": 3}[layout]
    offsets = (offset, 2) if reverse else (2, offset)
    source = (
        setup
        + f"""
backing = torch.ones(64, device=device)
a = backing.as_strided((2, 3), {stride!r}, {offsets[0]})
b = backing.as_strided((2, 3), {stride!r}, {offsets[1]})
infini.ops.mul_backward(x, x, x, a, b,
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


@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPES)
@pytest.mark.parametrize("layout", ("adjacent", "strided-disjoint"))
@pytest.mark.parametrize("reverse", (False, True), ids=("forward", "reverse"))
def test_mul_backward_disjoint_outputs_in_same_storage(
    layout,
    reverse,
    dtype,
    rtol,
    atol,
    device,
    implementation_index,
):
    inputs = _make_inputs((2, 3), dtype, device)
    backing = _backing(dtype, device)
    if layout == "adjacent":
        # Half-open ranges meet at the boundary; this is not overlap.
        grad_a = backing.as_strided((2, 3), (3, 1), 2)
        grad_b = backing.as_strided((2, 3), (3, 1), 8)
    else:
        grad_a = backing.as_strided((2, 3), (6, 2), 2)
        grad_b = backing.as_strided((2, 3), (6, 2), 18)
    if reverse:
        grad_a, grad_b = grad_b, grad_a
    expected = _reference_gradients(**inputs)
    expected_backing = backing.clone()
    for output, result in zip((grad_a, grad_b), expected):
        expected_backing.as_strided(
            output.shape,
            output.stride(),
            output.storage_offset(),
        ).copy_(result)

    _mul_backward(
        **inputs,
        grad_a=grad_a,
        grad_b=grad_b,
        implementation_index=implementation_index,
    )

    _assert_gradients(grad_a, grad_b, expected, rtol, atol)
    torch.testing.assert_close(backing, expected_backing, rtol=rtol, atol=atol)
