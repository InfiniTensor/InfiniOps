import json
import subprocess
import sys

import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream


# Format: (input_shape, mask_shape, grad_output_strides, mask_strides,
#          grad_input_strides, mask_mode)
_TEST_CASES = (
    pytest.param((17,), (17,), None, None, None, "alternating", id="one-dimensional"),
    pytest.param((3, 5), (3, 5), None, None, None, "all-false", id="all-false-mask"),
    pytest.param((3, 5), (3, 5), None, None, None, "all-true", id="all-true-mask"),
    pytest.param(
        (2, 3, 4),
        (2, 3, 4),
        None,
        None,
        None,
        "alternating",
        id="three-dimensional",
    ),
    pytest.param(
        (2, 3, 4),
        (2, 3, 4),
        (20, 6, 1),
        (18, 5, 1),
        (24, 7, 1),
        "alternating",
        id="strided-tensors",
    ),
    pytest.param(
        (2, 3),
        (3,),
        None,
        None,
        None,
        "alternating",
        id="column-mask-broadcast",
    ),
    pytest.param(
        (2, 3),
        (2, 1),
        None,
        None,
        None,
        "alternating",
        id="explicit-row-mask-broadcast",
    ),
    pytest.param(
        (2, 3, 4),
        (4,),
        None,
        None,
        None,
        "alternating",
        id="mask-broadcast-across-two-dims",
    ),
    pytest.param(
        (1, 3),
        (2, 3),
        None,
        None,
        None,
        "alternating",
        id="input-broadcast-reduction",
    ),
    pytest.param(
        (1, 3),
        (2, 3),
        None,
        None,
        None,
        "all-false",
        id="input-broadcast-all-false",
    ),
    pytest.param(
        (1, 3),
        (2, 3),
        None,
        None,
        None,
        "all-true",
        id="input-broadcast-all-true",
    ),
    pytest.param(
        (3,),
        (2, 3),
        None,
        None,
        None,
        "alternating",
        id="input-missing-leading-dim",
    ),
    pytest.param(
        (2, 1),
        (1, 3),
        None,
        None,
        None,
        "alternating",
        id="both-operands-broadcast",
    ),
    pytest.param(
        (2, 1, 4),
        (1, 3, 1),
        None,
        None,
        None,
        "alternating",
        id="middle-dim-reduction",
    ),
    pytest.param(
        (1, 3, 1),
        (2, 1, 4),
        None,
        None,
        None,
        "alternating",
        id="multiple-dim-reduction",
    ),
    pytest.param(
        (2, 3),
        (1, 2, 3),
        None,
        None,
        None,
        "alternating",
        id="mask-adds-leading-singleton",
    ),
    pytest.param(
        (1, 3),
        (2, 3),
        (8, 2),
        (7, 2),
        (9, 2),
        "alternating",
        id="strided-reduction",
    ),
    pytest.param(
        (2, 1),
        (1, 3),
        (8, 2),
        (8, 2),
        (3, 1),
        "alternating",
        id="strided-mutual-broadcast",
    ),
    pytest.param((), (), None, None, None, "all-false", id="scalar-all-false"),
    pytest.param((), (), None, None, None, "all-true", id="scalar-all-true"),
    pytest.param((2, 3), (), None, None, None, "all-false", id="scalar-mask-all-false"),
    pytest.param((2, 3), (), None, None, None, "all-true", id="scalar-mask-all-true"),
    pytest.param(
        (),
        (2, 3),
        None,
        None,
        None,
        "alternating",
        id="scalar-input-reduction",
    ),
    pytest.param((0,), (0,), None, None, None, "alternating", id="empty-vector"),
    pytest.param(
        (0, 3),
        (3,),
        None,
        None,
        None,
        "alternating",
        id="empty-output-with-small-mask",
    ),
    pytest.param(
        (1, 3),
        (0, 3),
        None,
        None,
        None,
        "alternating",
        id="empty-output-reduces-to-nonempty-input",
    ),
    pytest.param(
        (2, 0, 4),
        (1, 0, 1),
        None,
        None,
        None,
        "alternating",
        id="empty-middle-dim",
    ),
    pytest.param(
        (),
        (0, 3),
        None,
        None,
        None,
        "alternating",
        id="empty-output-reduces-to-scalar",
    ),
)

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "input_shape",
        "mask_shape",
        "grad_output_strides",
        "mask_strides",
        "grad_input_strides",
        "mask_mode",
    ),
    _TEST_CASES,
)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_masked_fill_backward(
    input_shape,
    mask_shape,
    grad_output_strides,
    mask_strides,
    grad_input_strides,
    mask_mode,
    implementation_index,
    dtype,
    device,
):
    output_shape = torch.broadcast_shapes(input_shape, mask_shape)
    grad_output = _empty(output_shape, grad_output_strides, dtype, device)
    grad_output.normal_()
    if tuple(output_shape) != tuple(input_shape):
        # Small signed integers keep all these sums exactly representable,
        # including in FP16/BF16, so reduction tests can use zero tolerance.
        values = torch.arange(grad_output.numel(), device=device) % 7 - 3
        grad_output.copy_(values.reshape(output_shape))

    mask = _make_mask(mask_shape, mask_strides, mask_mode, device)
    grad_input = _empty(input_shape, grad_input_strides, dtype, device)
    grad_input.fill_(17)

    return Payload(
        lambda *args: _masked_fill_backward(
            *args, implementation_index=implementation_index
        ),
        _torch_masked_fill_backward,
        (grad_output, mask, grad_input),
        {},
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_masked_fill_backward_known_reduction(implementation_index, dtype, device):
    grad_output = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=dtype, device=device)
    mask = torch.tensor([[False, True, False], [False, False, True]], device=device)
    grad_input = torch.full((1, 3), 17, dtype=dtype, device=device)

    _masked_fill_backward(
        grad_output, mask, grad_input, implementation_index=implementation_index
    )

    expected = torch.tensor([[5, 5, 3]], dtype=dtype, device=device)
    torch.testing.assert_close(grad_input, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("input_shape", ((2, 3), (1, 3)))
@pytest.mark.parametrize("all_true", (False, True))
def test_masked_fill_backward_nonfinite_gradients(
    implementation_index, dtype, device, input_shape, all_true
):
    grad_output = torch.tensor(
        [[float("nan"), float("inf"), 3], [4, -float("inf"), float("nan")]],
        dtype=dtype,
        device=device,
    )
    mask = torch.tensor([[True, True, False], [False, False, False]], device=device)
    if all_true:
        mask.fill_(True)
    grad_input = torch.full(input_shape, 17, dtype=dtype, device=device)
    expected = _torch_masked_fill_backward(
        grad_output, mask, torch.empty_like(grad_input)
    )

    _masked_fill_backward(
        grad_output, mask, grad_input, implementation_index=implementation_index
    )

    torch.testing.assert_close(grad_input, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("input_shape", ((2, 3), (1, 3)))
def test_masked_fill_backward_expanded_readonly_inputs(
    implementation_index, dtype, device, input_shape
):
    # Both inputs have zero strides from expand(); output has ordinary storage.
    grad_output = torch.tensor([[1, 2, 3]], dtype=dtype, device=device).expand(2, 3)
    mask = torch.tensor([[False], [True]], device=device).expand(2, 3)
    grad_input = torch.full(input_shape, 17, dtype=dtype, device=device)
    expected = _torch_masked_fill_backward(
        grad_output, mask, torch.empty_like(grad_input)
    )

    _masked_fill_backward(
        grad_output, mask, grad_input, implementation_index=implementation_index
    )

    torch.testing.assert_close(grad_input, expected, rtol=0, atol=0)


# Call the implementation directly: a Payload helper may clone arguments and
# destroy the storage overlap that these tests are intended to exercise.
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "alias_mode",
    ("exact", "partial", "strided", "reduced", "mask"),
)
def test_masked_fill_backward_overlapping_storage(
    implementation_index, dtype, device, alias_mode
):
    storage = torch.arange(24, dtype=dtype, device=device)
    grad_output = storage[:6].reshape(2, 3)
    mask = _make_mask((2, 3), None, "alternating", device)

    if alias_mode == "exact":
        grad_input = grad_output
    elif alias_mode == "partial":
        grad_input = storage[1:7].reshape(2, 3)
    elif alias_mode == "strided":
        grad_input = storage[1:13:2].reshape(2, 3)
    elif alias_mode == "reduced":
        grad_input = storage[1:4].reshape(1, 3)
    else:
        # Different dtypes can still refer to overlapping byte ranges.
        grad_input = storage[:6].reshape(2, 3)
        grad_input.fill_(17)
        mask = storage.view(torch.bool)[:6].reshape(2, 3)
        mask.copy_(_make_mask((2, 3), None, "alternating", device))
        grad_output = torch.arange(1, 7, dtype=dtype, device=device).reshape(2, 3)

    # Preserve the original values before the implementation writes its output.
    expected = _torch_masked_fill_backward(
        grad_output.detach().clone(),
        mask.detach().clone(),
        torch.empty(grad_input.shape, dtype=dtype, device=device),
    )

    _masked_fill_backward(
        grad_output, mask, grad_input, implementation_index=implementation_index
    )

    torch.testing.assert_close(grad_input, expected, rtol=0, atol=0)


# Format: (input_shape, mask_shape, supplied_grad_output_shape)
_INVALID_SHAPE_CASES = (
    pytest.param((2, 3), (2,), (2, 3), id="reject-implicit-row-alignment"),
    pytest.param((2, 3), (2, 2), (2, 3), id="reject-incompatible-mask-dimension"),
    pytest.param((2, 3), (2, 3), (3,), id="reject-missing-output-dimension"),
    pytest.param((2, 3), (1, 2, 3), (2, 3), id="reject-missing-leading-singleton"),
    pytest.param((1, 3), (3,), (2, 3), id="reject-unexplained-output-expansion"),
    pytest.param((2, 3), (2, 3), (1, 3), id="reject-smaller-output"),
    pytest.param((2, 3), (0, 3), (0, 3), id="reject-zero-versus-nonsingleton"),
)


@pytest.mark.parametrize(
    ("input_shape", "mask_shape", "output_shape"), _INVALID_SHAPE_CASES
)
def test_masked_fill_backward_rejects_invalid_shapes(
    implementation_index, device, input_shape, mask_shape, output_shape
):
    setup = f"device = torch.device({str(device)!r})\n"
    setup += f"grad_output = torch.ones({output_shape!r}, device=device)\n"
    setup += f"mask = torch.zeros({mask_shape!r}, dtype=torch.bool, device=device)\n"
    setup += f"grad_input = torch.empty({input_shape!r}, device=device)\n"
    probe = setup + "infini.ops.MaskedFillBackward(grad_output, mask, grad_input)\n"
    source = (
        setup
        + f"infini.ops.masked_fill_backward(grad_output, mask, grad_input, stream=get_stream(device), implementation_index={implementation_index})\n"
    )
    # Run the constructor probe first; isolate C++ assertions from pytest.
    prelude = (
        f"import sys; sys.path[:] = {json.dumps(sys.path)}\n"
        "import infini.ops\nimport torch\n"
        "from tests.utils import get_stream\n"
    )

    for code, expected_message in (
        (probe, None),
        (source, "broadcast shape of grad_input and mask"),
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


def _empty(shape, strides, dtype, device):
    if strides is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return torch.empty_strided(shape, strides, dtype=dtype, device=device)


def _make_mask(shape, strides, mode, device):
    mask = _empty(shape, strides, torch.bool, device)
    if mode == "all-false":
        mask.fill_(False)
    elif mode == "all-true":
        mask.fill_(True)
    else:
        values = torch.arange(mask.numel(), device=device).reshape(shape) % 3 == 0
        mask.copy_(values)
    return mask


def _masked_fill_backward(grad_output, mask, grad_input, *, implementation_index):
    infini.ops.masked_fill_backward(
        grad_output,
        mask,
        grad_input,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )
    return grad_input


def _torch_masked_fill_backward(grad_output, mask, grad_input):
    # Use forward + autograd as the oracle, rather than reimplementing the
    # masking/reduction formula under test. Snapshot inputs for alias safety.
    saved_grad = grad_output.detach().clone()
    saved_mask = mask.detach().clone()
    with torch.enable_grad():
        input_tensor = torch.zeros(
            grad_input.shape,
            dtype=grad_input.dtype,
            device=grad_input.device,
            requires_grad=True,
        )
        output = input_tensor.masked_fill(saved_mask, -7)
        (expected,) = torch.autograd.grad(output, input_tensor, grad_outputs=saved_grad)
    grad_input.copy_(expected)
    return grad_input
