import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Format: (shape, grad_output_strides, mask_strides, grad_input_strides, mask_mode)
_TEST_CASES = (
    pytest.param(
        (17,),
        None,
        None,
        None,
        "alternating",
        id="one-dimensional",
    ),
    pytest.param(
        (3, 5),
        None,
        None,
        None,
        "all-false",
        id="all-false-mask",
    ),
    pytest.param(
        (3, 5),
        None,
        None,
        None,
        "all-true",
        id="all-true-mask",
    ),
    pytest.param(
        (2, 3, 4),
        None,
        None,
        None,
        "alternating",
        id="three-dimensional",
    ),
    pytest.param(
        (2, 3, 4),
        (20, 6, 1),
        (18, 5, 1),
        (24, 7, 1),
        "alternating",
        id="strided-tensors",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "shape",
        "grad_output_strides",
        "mask_strides",
        "grad_input_strides",
        "mask_mode",
    ),
    _TEST_CASES,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 0, 0),
        (torch.float16, 0, 0),
        (torch.bfloat16, 0, 0),
    ),
)
def test_masked_fill_backward(
    shape,
    grad_output_strides,
    mask_strides,
    grad_input_strides,
    mask_mode,
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
    mask = _make_mask(shape, mask_strides, mask_mode, device)
    grad_input = empty_strided(
        shape,
        grad_input_strides,
        dtype=dtype,
        device=device,
    )
    grad_input.fill_(17)

    return Payload(
        lambda *args: _masked_fill_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_masked_fill_backward,
        (grad_output, mask, grad_input),
        {},
        rtol=rtol,
        atol=atol,
    )


def _make_mask(shape, strides, mode, device):
    mask = empty_strided(shape, strides, dtype=torch.bool, device=device)

    if mode == "all-false":
        mask.fill_(False)
    elif mode == "all-true":
        mask.fill_(True)
    else:
        values = torch.arange(mask.numel(), device=device).reshape(shape) % 3 == 0
        mask.copy_(values)

    return mask


def _masked_fill_backward(
    grad_output,
    mask,
    grad_input,
    *,
    implementation_index,
):
    infini.ops.masked_fill_backward(
        grad_output,
        mask,
        grad_input,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )

    return grad_input


def _torch_masked_fill_backward(grad_output, mask, grad_input):
    grad_input.copy_(grad_output.masked_fill(mask, 0))

    return grad_input
