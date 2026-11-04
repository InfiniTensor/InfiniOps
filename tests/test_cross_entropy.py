import infini.ops
import pytest
import torch

from tests.utils import (
    Payload,
    empty_strided,
    get_stream,
    randn_strided,
    randint_strided,
)


# Format: (input_shape, input_strides, target_strides)
# CrossEntropy follows the PyTorch convention: dimension 1 is the class dimension.
_TEST_CASES = (
    pytest.param(
        (4, 7),
        None,
        None,
        id="two-dimensional",
    ),
    pytest.param(
        (2, 3, 5),
        None,
        None,
        id="three-dimensional",
    ),
    pytest.param(
        (2, 3, 5),
        (30, 7, 1),
        None,
        id="non-contiguous-input",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, input_strides, target_strides",
    _TEST_CASES,
)
@pytest.mark.parametrize("use_ignore_index", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_cross_entropy(
    input_shape,
    input_strides,
    target_strides,
    use_ignore_index,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    input = randn_strided(
        input_shape,
        input_strides,
        dtype=dtype,
        device=device,
    )
    target_shape = (input_shape[0],) + input_shape[2:]
    num_classes = input_shape[1]
    target = randint_strided(
        0,
        num_classes,
        target_shape,
        target_strides,
        dtype=torch.int64,
        device=device,
    )
    if use_ignore_index:
        target.reshape(-1)[0] = -100
    out = empty_strided((), None, dtype=dtype, device=device)
    out.fill_(17)

    return Payload(
        lambda *args: _cross_entropy(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_cross_entropy,
        (input, target, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _cross_entropy(input, target, out, *, implementation_index):
    infini.ops.cross_entropy(
        input,
        target,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_cross_entropy(input, target, out):
    out.copy_(
        torch.nn.functional.cross_entropy(
            input,
            target,
            reduction="mean",
            ignore_index=-100,
        )
    )

    return out
