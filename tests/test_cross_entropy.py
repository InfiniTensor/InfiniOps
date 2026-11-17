import math

import infini.ops
import pytest
import torch
import torch.nn.functional as F

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_DTYPES = (torch.float32, torch.float64, torch.float16, torch.bfloat16)
_REDUCTIONS = ("none", "mean", "sum")
# Dimension 0 is the class dimension for (C,); otherwise it is dimension 1.
_SHAPES = (
    pytest.param((7,), False, id="unbatched"),
    pytest.param((1, 7), False, id="batch-one"),
    pytest.param((4, 7), False, id="two-dimensional"),
    pytest.param((2, 3, 5), False, id="three-dimensional"),
    pytest.param((2, 3, 2, 4), False, id="four-dimensional"),
    pytest.param((3, 1), False, id="one-class"),
    pytest.param((7,), True, id="unbatched-strided"),
    pytest.param((2, 3, 5), True, id="all-tensors-strided"),
)


def _tolerance(dtype):
    return {
        torch.float64: (1e-10, 1e-10),
        torch.float32: (1e-5, 1e-5),
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (1e-2, 1e-2),
    }[dtype]


def _strides(shape, strided):
    if not strided or not shape:
        return None
    result = []
    step = 2
    for size in reversed(shape):
        result.append(step)
        step *= max(size, 1)
    return tuple(reversed(result))


def _class_dim(shape):
    return 0 if len(shape) == 1 else 1


def _loss_shape(shape):
    dim = _class_dim(shape)
    return shape[:dim] + shape[dim + 1 :]


def _promote_low_precision(tensor):
    if tensor is not None and tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.float()
    return tensor


def _result_dtype(input, target, weight):
    dtype = input.dtype
    if target.is_floating_point():
        dtype = torch.promote_types(dtype, target.dtype)
        if weight is not None:
            dtype = torch.promote_types(dtype, weight.dtype)
    return dtype


def _expected(input, target, weight, ignore_index, reduction, label_smoothing):
    # FP16/BF16 reference uses FP32 intermediates, not same-dtype Torch CE.
    target = (
        _promote_low_precision(target) if target.is_floating_point() else target.long()
    )
    return F.cross_entropy(
        _promote_low_precision(input),
        target,
        weight=_promote_low_precision(weight),
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


def _cross_entropy(
    input,
    target,
    weight,
    ignore_index,
    reduction,
    label_smoothing,
    out,
    *,
    implementation_index,
):
    # Requires the binding for the new canonical C++ argument order.
    infini.ops.cross_entropy(
        input,
        target,
        weight,
        ignore_index,
        reduction,
        label_smoothing,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )
    return out


def _torch_cross_entropy(
    input,
    target,
    weight,
    ignore_index,
    reduction,
    label_smoothing,
    out,
):
    out.copy_(
        _expected(
            input,
            target,
            weight,
            ignore_index,
            reduction,
            label_smoothing,
        )
    )
    return out


def _data(shape, probability_target, weighted, strided, dtype, device):
    input = randn_strided(
        shape,
        _strides(shape, strided),
        dtype=dtype,
        device=device,
    )
    dim = _class_dim(shape)
    classes = shape[dim]
    if probability_target:
        target = empty_strided(
            shape,
            _strides(shape, strided),
            dtype=dtype,
            device=device,
        )
        target.copy_(torch.randn(shape, device=device).softmax(dim=dim))
    else:
        target_shape = _loss_shape(shape)
        target = empty_strided(
            target_shape,
            _strides(target_shape, strided),
            dtype=torch.int64,
            device=device,
        )
        target.copy_(torch.randint(classes, target_shape, device=device))
    weight = None
    if weighted:
        weight = empty_strided(
            (classes,),
            _strides((classes,), strided),
            dtype=dtype,
            device=device,
        )
        weight.copy_(torch.linspace(0.25, 2.0, classes, device=device))
    return input, target, weight


def _out(input, target, weight, reduction, strided=False):
    shape = _loss_shape(tuple(input.shape)) if reduction == "none" else ()
    out = empty_strided(
        shape,
        _strides(shape, strided),
        dtype=_result_dtype(input, target, weight),
        device=input.device,
    )
    out.fill_(17)
    return out


def _check(
    input,
    target,
    weight,
    ignore_index,
    reduction,
    label_smoothing,
    implementation_index,
    out=None,
):
    if out is None:
        out = _out(input, target, weight, reduction)
    # Compute and detach the reference storage BEFORE the backend can write
    # an aliased output. Merely assigning another Tensor variable is not a copy.
    expected = (
        _expected(
            input,
            target,
            weight,
            ignore_index,
            reduction,
            label_smoothing,
        )
        .to(out.dtype)
        .clone()
    )
    _cross_entropy(
        input,
        target,
        weight,
        ignore_index,
        reduction,
        label_smoothing,
        out,
        implementation_index=implementation_index,
    )
    rtol, atol = _tolerance(out.dtype)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol, equal_nan=True)
    return out


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("shape,strided", _SHAPES)
@pytest.mark.parametrize("probability_target", (False, True))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("label_smoothing", (0.0, 0.2, 1.0))
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cross_entropy(
    shape,
    strided,
    probability_target,
    weighted,
    reduction,
    label_smoothing,
    implementation_index,
    dtype,
    device,
):
    input, target, weight = _data(
        shape,
        probability_target,
        weighted,
        strided,
        dtype,
        device,
    )
    out = _out(input, target, weight, reduction, strided)
    rtol, atol = _tolerance(out.dtype)
    return Payload(
        lambda *args: _cross_entropy(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_cross_entropy,
        (input, target, weight, -100, reduction, label_smoothing, out),
        {},
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("pattern", ("partial", "one-valid", "all"))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("label_smoothing", (0.0, 0.2))
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cross_entropy_ignored_targets(
    pattern,
    weighted,
    reduction,
    label_smoothing,
    implementation_index,
    dtype,
    device,
):
    input, target, weight = _data((2, 3, 5), False, weighted, True, dtype, device)
    if pattern == "partial":
        target[0, ::2] = -100
    else:
        target.fill_(-100)
        if pattern == "one-valid":
            target[1, 3] = 1
    out = _check(
        input,
        target,
        weight,
        -100,
        reduction,
        label_smoothing,
        implementation_index,
    )
    if pattern == "all":
        if reduction == "mean":
            assert torch.isnan(out).item()
        else:
            assert torch.count_nonzero(out).item() == 0
    elif reduction == "none":
        assert torch.count_nonzero(out[target == -100]).item() == 0


@pytest.mark.parametrize("ignore_index", (-7, 1, 255))
@pytest.mark.parametrize("reduction", _REDUCTIONS)
def test_cross_entropy_custom_ignore_index(
    ignore_index,
    reduction,
    implementation_index,
    device,
):
    input = torch.zeros((3, 3), device=device)
    target = torch.tensor([ignore_index, 0, 2], device=device)
    out = _check(
        input, target, None, ignore_index, reduction, 0.0, implementation_index
    )
    expected = {
        "none": [0.0, math.log(3), math.log(3)],
        "mean": math.log(3),
        "sum": 2 * math.log(3),
    }[reduction]
    torch.testing.assert_close(out, torch.tensor(expected, device=device))


@pytest.mark.parametrize("target_dtype", (torch.int32, torch.uint8))
def test_cross_entropy_integer_target_extensions(
    target_dtype,
    implementation_index,
    device,
):
    input = torch.zeros((3, 3), device=device)
    target = torch.tensor([0, 1, 255], dtype=target_dtype, device=device)
    _check(input, target, None, 255, "mean", 0.2, implementation_index)


@pytest.mark.parametrize("probability_target", (False, True))
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("all_zero", (False, True))
def test_cross_entropy_zero_weights(
    probability_target,
    reduction,
    all_zero,
    implementation_index,
    device,
):
    input, target, _ = _data(
        (3, 3), probability_target, False, False, torch.float32, device
    )
    if not probability_target:
        target.copy_(torch.tensor([0, 1, 2], device=device))
    weight = torch.tensor([0.0, 0.0 if all_zero else 2.0, 0.0], device=device)
    out = _check(input, target, weight, -100, reduction, 0.2, implementation_index)
    if all_zero:
        if not probability_target and reduction == "mean":
            assert torch.isnan(out).item()
        else:
            assert torch.count_nonzero(out).item() == 0


def test_cross_entropy_probability_zero_row(implementation_index, device):
    input = torch.zeros((2, 3), device=device)
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=device)
    # A zero probability row does not remove that row from the mean divisor.
    out = _check(input, target, None, -100, "mean", 0.0, implementation_index)
    torch.testing.assert_close(out, torch.tensor(math.log(3) / 2, device=device))


@pytest.mark.parametrize("probability_target", (False, True))
def test_cross_entropy_weighted_mean_divisor(
    probability_target,
    implementation_index,
    device,
):
    input = torch.zeros((2, 2), device=device)
    target = torch.tensor([0, 1], device=device)
    weight = torch.tensor([1.0, 3.0], device=device)
    if probability_target:
        target = F.one_hot(target, num_classes=2).float()
    out = _check(input, target, weight, -100, "mean", 0.0, implementation_index)
    # Same one-hot labels, different upstream mean denominators: 4 vs 2.
    expected = math.log(2) * (2 if probability_target else 1)
    torch.testing.assert_close(out, torch.tensor(expected, device=device))


@pytest.mark.parametrize("probability_target", (False, True))
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("label_smoothing", (0.0, 0.2, 1.0))
def test_cross_entropy_smoothing_analytic(
    probability_target,
    reduction,
    label_smoothing,
    implementation_index,
    device,
):
    # Each row assigns probability 3/4 to its correct class and 1/4 to the
    # other class. Calculate the answer without calling Torch CE.
    input = torch.tensor(
        [[math.log(3), 0.0], [0.0, math.log(3)]], dtype=torch.float64, device=device
    )
    target = torch.tensor([0, 1], device=device)
    weight = torch.tensor([1.0, 3.0], dtype=torch.float64, device=device)
    if probability_target:
        target = F.one_hot(target, num_classes=2).double()
    a = 1 - label_smoothing / 2
    b = label_smoothing / 2
    losses = [
        -a * math.log(0.75) - 3 * b * math.log(0.25),
        -3 * a * math.log(0.75) - b * math.log(0.25),
    ]
    expected = torch.tensor(losses, dtype=torch.float64, device=device)
    if reduction == "sum":
        expected = expected.sum()
    elif reduction == "mean":
        expected = expected.sum() / (2 if probability_target else 4)
    out = _check(
        input, target, weight, -100, reduction, label_smoothing, implementation_index
    )
    torch.testing.assert_close(out, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("reduction", _REDUCTIONS)
def test_cross_entropy_unbatched_ignored(reduction, implementation_index, device):
    input = torch.zeros((3,), device=device)
    target = torch.tensor(-100, device=device)
    out = _check(input, target, None, -100, reduction, 0.2, implementation_index)
    if reduction == "mean":
        assert torch.isnan(out).item()
    else:
        assert out.item() == 0


@pytest.mark.parametrize("shape", ((0, 3), (2, 3, 0)))
@pytest.mark.parametrize("probability_target", (False, True))
@pytest.mark.parametrize("reduction", _REDUCTIONS)
def test_cross_entropy_empty(
    shape,
    probability_target,
    reduction,
    implementation_index,
    device,
):
    input, target, weight = _data(
        shape, probability_target, True, False, torch.float32, device
    )
    out = _check(input, target, weight, -100, reduction, 0.2, implementation_index)
    if reduction == "mean":
        assert torch.isnan(out).item()
    elif reduction == "sum":
        assert out.item() == 0
    else:
        assert out.numel() == 0


@pytest.mark.parametrize("probability_target", (False, True))
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_cross_entropy_extreme_logits(
    probability_target,
    reduction,
    dtype,
    implementation_index,
    device,
):
    input = torch.tensor(
        [
            [1000.0, -1000.0, 0.0],
            [-1000.0, 1000.0, 0.0],
            [1000.0, 1000.0, 1000.0],
            [-1000.0, -1000.0, -1000.0],
        ],
        dtype=dtype,
        device=device,
    )
    target = torch.tensor([1, 1, 0, 2], device=device)
    if probability_target:
        target = F.one_hot(target, num_classes=3).to(dtype)
    out = _check(input, target, None, -100, reduction, 0.0, implementation_index)
    losses = torch.tensor(
        [2000.0, 0.0, math.log(3), math.log(3)], dtype=torch.float64, device=device
    )
    expected = (
        losses
        if reduction == "none"
        else (losses.mean() if reduction == "mean" else losses.sum())
    )
    rtol, atol = _tolerance(dtype)
    assert torch.isfinite(out).all().item()
    torch.testing.assert_close(out, expected.to(dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("count", (32768, 65504, 65505, 65520, 65536, 131072))
@pytest.mark.parametrize("classes", (2, 4))
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_cross_entropy_long_sequence(
    count,
    classes,
    weighted,
    dtype,
    implementation_index,
    device,
):
    # Shape (1,C,L) deliberately exercises the sequence path. With a constant
    # class weight of 2, even count=32768 has total_weight > FP16's maximum.
    input = torch.zeros((1, classes, count + 17), dtype=dtype, device=device)
    target = torch.full((1, count + 17), -100, dtype=torch.int64, device=device)
    target[:, :count] = 0
    weight = (
        torch.full((classes,), 2.0, dtype=dtype, device=device) if weighted else None
    )
    out = _check(input, target, weight, -100, "mean", 0.0, implementation_index)
    # Independent mathematical oracle: each prediction is uniform.
    rtol, atol = _tolerance(dtype)
    assert torch.isfinite(out).item()
    torch.testing.assert_close(
        out,
        torch.tensor(math.log(classes), dtype=dtype, device=device),
        rtol=rtol,
        atol=atol,
    )


def test_cross_entropy_sum_output_overflow(implementation_index, device):
    # FP32 intermediates cannot make a genuinely unrepresentable FP16 output
    # finite. This must remain +inf, unlike the finite long-sequence mean.
    input = torch.zeros((1, 4, 65536), dtype=torch.float16, device=device)
    target = torch.zeros((1, 65536), dtype=torch.int64, device=device)
    out = _check(input, target, None, -100, "sum", 0.0, implementation_index)
    assert torch.isposinf(out).item()


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_cross_entropy_long_probability_target(dtype, implementation_index, device):
    input = torch.zeros((1, 4, 65536), dtype=dtype, device=device)
    target = torch.full_like(input, 0.25)
    out = _check(input, target, None, -100, "mean", 0.2, implementation_index)
    rtol, atol = _tolerance(dtype)
    assert torch.isfinite(out).item()
    torch.testing.assert_close(
        out,
        torch.tensor(math.log(4), dtype=dtype, device=device),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    "input_dtype,target_dtype",
    (
        (torch.float32, torch.float64),
        (torch.float16, torch.float32),
        (torch.float16, torch.bfloat16),
    ),
)
def test_cross_entropy_probability_dtype_promotion(
    input_dtype,
    target_dtype,
    implementation_index,
    device,
):
    input, target, weight = _data((2, 3), True, True, True, input_dtype, device)
    target = target.to(target_dtype)
    _check(input, target, weight, -100, "none", 0.2, implementation_index)


def test_cross_entropy_probability_weight_dtype_promotion(implementation_index, device):
    input, target, weight = _data((2, 3), True, True, True, torch.float32, device)
    _check(input, target, weight.double(), -100, "mean", 0.2, implementation_index)


@pytest.mark.parametrize("probability_target", (False, True))
@pytest.mark.parametrize("reduction", _REDUCTIONS)
def test_cross_entropy_transposed_layout(
    probability_target,
    reduction,
    implementation_index,
    device,
):
    input = torch.randn((3, 2, 5), device=device).transpose(0, 1)
    if probability_target:
        target = torch.randn((3, 2, 5), device=device).softmax(0).transpose(0, 1)
    else:
        target = (torch.arange(10, device=device).reshape(5, 2) % 3).t()
        target[0, 0] = -100
    weight = torch.tensor([0.5, 17.0, 2.0, 17.0, 3.0], device=device)[::2]
    out = (
        torch.full((5, 2), 17.0, device=device).t()
        if reduction == "none"
        else torch.full((), 17.0, device=device)
    )
    _check(input, target, weight, -100, reduction, 0.2, implementation_index, out=out)


@pytest.mark.parametrize(
    "alias",
    (
        "input-full",
        "input-partial",
        "target-full",
        "target-partial",
        "weight",
    ),
)
@pytest.mark.parametrize("reduction", ("none", "mean"))
def test_cross_entropy_aliasing(alias, reduction, implementation_index, device):
    full = alias in ("input-full", "target-full")
    shape = (1,) if full and reduction == "mean" else ((2, 1) if full else (2, 3))
    probability_target = alias.startswith("target")
    input, target, weight = _data(
        shape, probability_target, True, True, torch.float32, device
    )
    if alias == "weight":
        out = weight[:2] if reduction == "none" else weight[1]
    else:
        source = target if probability_target else input
        if full:
            out = source.squeeze(1) if reduction == "none" else source[0]
        else:
            out = source[:, 0] if reduction == "none" else source[0, 1]
    _check(input, target, weight, -100, reduction, 0.2, implementation_index, out=out)


def test_cross_entropy_strided_output_guards(implementation_index, device):
    input, target, weight = _data((2, 3), False, True, True, torch.float32, device)
    backing = torch.full((7,), 17.0, device=device)
    out = backing[1:5:2]
    _check(input, target, weight, -100, "none", 0.2, implementation_index, out=out)
    untouched = torch.tensor([0, 2, 4, 5, 6], device=device)
    torch.testing.assert_close(
        backing[untouched], torch.full((5,), 17.0, device=device)
    )
