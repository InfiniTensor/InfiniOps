import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream


# Format: (rows, vocab_size_local, vocab_size_original, label_smoothing,
#          grad_output_mode, grad_input_strides)
_TEST_CASES = (
    pytest.param(
        3,
        4,
        6,
        0.0,
        "scalar",
        None,
        id="scalar-grad-no-smoothing",
    ),
    pytest.param(
        3,
        4,
        8,
        0.1,
        "per-row",
        None,
        id="per-row-grad-with-smoothing",
    ),
    pytest.param(
        4,
        5,
        8,
        0.2,
        "scalar",
        (28, 5),
        id="strided-grad-input",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "rows",
        "vocab_size_local",
        "vocab_size_original",
        "label_smoothing",
        "grad_output_mode",
        "grad_input_strides",
    ),
    _TEST_CASES,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 2e-3, 2e-3),
        (torch.bfloat16, 2e-2, 2e-2),
    ),
)
def test_vocab_parallel_cross_entropy_backward(
    rows,
    vocab_size_local,
    vocab_size_original,
    label_smoothing,
    grad_output_mode,
    grad_input_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    softmax_local = torch.rand(
        (rows, vocab_size_local),
        dtype=dtype,
        device=device,
    ).softmax(dim=-1)
    target_mask = torch.tensor(
        [0, 1, 0, 0][:rows],
        dtype=dtype,
        device=device,
    )
    masked_target = torch.tensor(
        [1, 2, 0, 3][:rows],
        dtype=torch.int64,
        device=device,
    )
    valid_mask_local = torch.tensor(
        [1, 1, 0, 1, 1][:vocab_size_local],
        dtype=dtype,
        device=device,
    )
    if grad_output_mode == "scalar":
        grad_output = torch.tensor(1.25, dtype=dtype, device=device)
    else:
        grad_output = torch.linspace(
            0.5,
            1.5,
            rows,
            dtype=dtype,
            device=device,
        )
    grad_input = empty_strided(
        (rows, vocab_size_local),
        grad_input_strides,
        dtype=dtype,
        device=device,
    )
    grad_input.fill_(17)

    return Payload(
        lambda *args: _vocab_parallel_cross_entropy_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_vocab_parallel_cross_entropy_backward,
        (
            grad_output,
            softmax_local,
            target_mask,
            masked_target,
            valid_mask_local,
            vocab_size_local,
            vocab_size_original,
            label_smoothing,
            grad_input,
        ),
        {},
        rtol=rtol,
        atol=atol,
    )


def _vocab_parallel_cross_entropy_backward(
    grad_output,
    softmax_local,
    target_mask,
    masked_target,
    valid_mask_local,
    vocab_size_local,
    vocab_size_original,
    label_smoothing,
    grad_input,
    *,
    implementation_index,
):
    infini.ops.vocab_parallel_cross_entropy_backward(
        grad_output,
        softmax_local,
        target_mask,
        masked_target,
        valid_mask_local,
        vocab_size_local,
        vocab_size_original,
        label_smoothing,
        grad_input,
        stream=get_stream(grad_output.device),
        implementation_index=implementation_index,
    )

    return grad_input


def _torch_vocab_parallel_cross_entropy_backward(
    grad_output,
    softmax_local,
    target_mask,
    masked_target,
    valid_mask_local,
    vocab_size_local,
    vocab_size_original,
    label_smoothing,
    grad_input,
):
    probabilities = softmax_local.detach().to(device="cpu", dtype=torch.float64)
    rows = probabilities.shape[0]
    dloss = grad_output.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if dloss.numel() == 1:
        dloss = dloss.expand(rows)

    valid_rows = 1 - target_mask.detach().to(device="cpu", dtype=torch.float64)
    valid_columns = (
        valid_mask_local.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(1, vocab_size_local)
    )
    distribution = (label_smoothing / vocab_size_original) * valid_columns.expand_as(
        probabilities
    ).clone()
    target = masked_target.detach().to(device="cpu", dtype=torch.int64).reshape(rows, 1)
    distribution.scatter_add_(
        1, target, ((1 - label_smoothing) * valid_rows).reshape(rows, 1)
    )
    gradient = (probabilities - distribution) * dloss.reshape(rows, 1) * valid_columns
    grad_input.copy_(gradient)

    return grad_input


@pytest.mark.parametrize(
    "dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64)
)
@pytest.mark.parametrize("smoothing", (1e-4, 1e-8))
@pytest.mark.parametrize("alias", (False, True))
def test_vocab_parallel_cross_entropy_backward_small_smoothing(
    dtype, smoothing, alias, device, implementation_index
):
    probabilities = torch.tensor([[1.0, 0.0]], dtype=dtype, device=device)
    upstream = torch.tensor(60000.0, dtype=dtype, device=device)
    target_mask = torch.zeros(1, dtype=dtype, device=device)
    target = torch.zeros(1, dtype=torch.int64, device=device)
    valid = torch.ones(2, dtype=dtype, device=device)
    output = probabilities if alias else torch.empty_like(probabilities)
    expected = torch.empty_like(probabilities)
    _torch_vocab_parallel_cross_entropy_backward(
        upstream, probabilities, target_mask, target, valid, 2, 2, smoothing, expected
    )
    _vocab_parallel_cross_entropy_backward(
        upstream,
        probabilities,
        target_mask,
        target,
        valid,
        2,
        2,
        smoothing,
        output,
        implementation_index=implementation_index,
    )
    assert output[0, 0].item() > 0
    assert output[0, 1].item() < 0
    tolerance = {
        torch.float16: 1e-3,
        torch.bfloat16: 1e-2,
        torch.float32: 1e-6,
        torch.float64: 1e-7,
    }[dtype]
    torch.testing.assert_close(output, expected, rtol=tolerance, atol=0)
