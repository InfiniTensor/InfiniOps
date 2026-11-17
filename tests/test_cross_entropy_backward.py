import infini.ops
import pytest
import torch

from tests.utils import (
    Payload,
    empty_strided,
    get_stream,
    randint_strided,
    randn_strided,
)


# Format: (input_shape, input_strides, target_strides, grad_input_strides)
# CrossEntropy uses the PyTorch convention: dimension 1 is the class dimension.
_TEST_CASES = (
    pytest.param(
        (4, 7),
        None,
        None,
        None,
        id="two-dimensional",
    ),
    pytest.param(
        (2, 3, 5),
        None,
        None,
        None,
        id="three-dimensional",
    ),
    pytest.param(
        (2, 3, 5),
        (30, 7, 1),
        None,
        (35, 8, 1),
        id="strided-input-and-output",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, input_strides, target_strides, grad_input_strides",
    _TEST_CASES,
)
@pytest.mark.parametrize("use_ignore_index", (False, True))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_cross_entropy_backward(
    input_shape,
    input_strides,
    target_strides,
    grad_input_strides,
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
    target = randint_strided(
        0,
        input_shape[1],
        (input_shape[0],) + input_shape[2:],
        target_strides,
        dtype=torch.int64,
        device=device,
    )
    if use_ignore_index:
        target.reshape(-1)[0] = -100
    grad_output = torch.randn((), dtype=dtype, device=device)
    grad_input = empty_strided(
        input_shape,
        grad_input_strides,
        dtype=dtype,
        device=device,
    )
    grad_input.fill_(17)

    return Payload(
        lambda *args: _cross_entropy_backward(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_cross_entropy_backward,
        (input, target, grad_output, grad_input),
        {},
        rtol=rtol,
        atol=atol,
    )


def _cross_entropy_backward(
    input,
    target,
    grad_output,
    grad_input,
    *,
    implementation_index,
):
    infini.ops.cross_entropy_backward(
        input,
        target,
        grad_output,
        None,
        -100,
        "mean",
        0.0,
        grad_input,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return grad_input


def _torch_cross_entropy_backward(input, target, grad_output, grad_input):
    logits = input.detach().clone().requires_grad_(True)
    loss = torch.nn.functional.cross_entropy(
        logits,
        target,
        reduction="mean",
        ignore_index=-100,
    )
    gradient = torch.autograd.grad(loss, logits, grad_output)[0]
    grad_input.copy_(gradient)

    return grad_input


# Forward API coverage is shared so backward stays aligned with every attribute.
@pytest.mark.parametrize(
    "shape,strided", (((7,), False), ((4, 7), False), ((2, 3, 5), True))
)
@pytest.mark.parametrize("probability_target", (False, True))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("reduction", ("none", "sum", "mean"))
@pytest.mark.parametrize("smoothing", (0.0, 0.2, 1.0))
@pytest.mark.parametrize(
    "dtype", (torch.float64, torch.float32, torch.float16, torch.bfloat16)
)
def test_cross_entropy_backward_full_contract(
    shape,
    strided,
    probability_target,
    weighted,
    reduction,
    smoothing,
    dtype,
    device,
    implementation_index,
):
    from tests.test_cross_entropy import _data

    input, target, weight = _data(
        shape, probability_target, weighted, strided, dtype, device
    )
    _check_full_backward(
        input, target, weight, -100, reduction, smoothing, implementation_index
    )


def _check_full_backward(
    input,
    target,
    weight,
    ignore_index,
    reduction,
    smoothing,
    implementation_index,
    out=None,
):
    from tests.test_cross_entropy import _expected, _result_dtype, _tolerance

    with torch.enable_grad():
        logits = input.detach().clone().requires_grad_(True)
        loss = _expected(
            logits, target.detach().clone(), weight, ignore_index, reduction, smoothing
        )
        upstream = torch.full_like(
            loss, 0.75, dtype=_result_dtype(input, target, weight)
        )
        expected = torch.autograd.grad(loss, logits, upstream)[0].detach().clone()

    if out is None:
        out = torch.empty_like(input)

    infini.ops.cross_entropy_backward(
        input,
        target,
        upstream,
        weight,
        ignore_index,
        reduction,
        smoothing,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )
    rtol, atol = _tolerance(input.dtype)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol, equal_nan=True)
    return out


@pytest.mark.parametrize("count", (65504, 65536, 131072))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("mode", ("ordinary", "all-ignore", "one-valid", "extreme"))
def test_cross_entropy_backward_long_sequence(
    count, dtype, mode, device, implementation_index
):
    input = torch.zeros((1, 4, count), dtype=dtype, device=device)
    target = torch.zeros((1, count), dtype=torch.int64, device=device)

    if mode == "all-ignore":
        target.fill_(-100)
    elif mode == "one-valid":
        target.fill_(-100)
        target[0, 0] = 0
    elif mode == "extreme":
        input[:, 1, :] = 1000

    out = _check_full_backward(
        input, target, None, -100, "mean", 0.0, implementation_index
    )

    if mode == "ordinary":
        expected = torch.full_like(input, 0.75 * 0.25 / count)
        expected[:, 0, :] = -0.75 * 0.75 / count
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    if mode == "all-ignore":
        assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("smoothing", (0.0, 0.2))
@pytest.mark.parametrize("reduction", ("none", "sum", "mean"))
def test_cross_entropy_backward_zero_weights(
    smoothing, reduction, device, implementation_index
):
    input = torch.randn((3, 4), device=device)
    target = torch.tensor([-7, 1, 2], device=device)
    weight = torch.zeros(4, device=device)
    _check_full_backward(
        input, target, weight, -7, reduction, smoothing, implementation_index
    )


@pytest.mark.parametrize("target_dtype", (torch.float16, torch.float32, torch.float64))
@pytest.mark.parametrize("weight_dtype", (torch.float32, torch.float64))
def test_cross_entropy_backward_probability_promotion(
    target_dtype,
    weight_dtype,
    device,
    implementation_index,
):
    input = torch.randn((2, 3), device=device)
    target = torch.randn((2, 3), device=device).softmax(1).to(target_dtype)
    weight = torch.tensor([0.5, 1.0, 2.0], dtype=weight_dtype, device=device)
    _check_full_backward(input, target, weight, -100, "mean", 0.2, implementation_index)


@pytest.mark.parametrize("ignore_index", (-7, 1, 255))
@pytest.mark.parametrize("reduction", ("none", "mean", "sum"))
@pytest.mark.parametrize("smoothing", (0.0, 0.2))
def test_cross_entropy_backward_custom_ignore_index(
    ignore_index,
    reduction,
    smoothing,
    device,
    implementation_index,
):
    input = torch.randn((2, 3), device=device)
    target = torch.tensor([ignore_index, 2], device=device)
    weight = torch.tensor([0.5, 1.0, 2.0], device=device)
    _check_full_backward(
        input, target, weight, ignore_index, reduction, smoothing, implementation_index
    )


@pytest.mark.parametrize("alias", ("input", "upstream", "weight"))
def test_cross_entropy_backward_alias(alias, device, implementation_index):
    input = torch.randn((1, 3), device=device)
    target = torch.tensor([1], device=device)
    weight = torch.tensor([0.5, 1.0, 2.0], device=device)
    # Input/output and weight/output may share memory; snapshot the oracle first.
    out = input if alias == "input" else weight.view(1, 3)

    if alias == "upstream":
        upstream = out.reshape(-1)[:1].view(())
        upstream.fill_(0.75)
        with torch.enable_grad():
            logits = input.detach().clone().requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(logits, target, weight=weight)
            expected = torch.autograd.grad(loss, logits, upstream.clone())[0]
        infini.ops.cross_entropy_backward(
            input,
            target,
            upstream,
            weight,
            -100,
            "mean",
            0.0,
            out,
            stream=get_stream(device),
            implementation_index=implementation_index,
        )
        torch.testing.assert_close(out, expected)
    else:
        _check_full_backward(
            input, target, weight, -100, "mean", 0.2, implementation_index, out
        )
