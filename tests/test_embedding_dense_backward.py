import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


# Dense-only interface: there is deliberately no sparse argument.
# (indices, num_weights, embedding_dim, indices_strides,
#  grad_output_strides, grad_weight_strides, padding_idx, scale_grad_by_freq)
_TEST_CASES = (
    pytest.param(
        (1, 2, 1, 4, 2, 1), 6, 4, None, None, None, -1, False, id="repeated-indices"
    ),
    pytest.param(
        ((0, 1, 0), (2, 1, 3)),
        5,
        7,
        (5, 1),
        (32, 8, 1),
        (9, 1),
        1,
        False,
        id="padding-and-strides",
    ),
    pytest.param(
        (2, 2, 2, 3, 3, 4),
        7,
        5,
        None,
        None,
        None,
        -1,
        True,
        id="scale-grad-by-frequency",
    ),
    pytest.param(
        (1, 2, 1, 1, 2, 4),
        6,
        5,
        (2,),
        (13, 2),
        (12, 2),
        1,
        True,
        id="padding-with-frequency-and-strides",
    ),
    pytest.param((), 4, 3, None, None, None, -1, False, id="empty"),
    pytest.param(
        (),
        4,
        3,
        (2,),
        (7, 2),
        (8, 2),
        0,
        True,
        id="empty-with-padding-frequency-and-strides",
    ),
    pytest.param(
        ((), ()), 4, 3, None, None, None, -1, True, id="multidimensional-empty"
    ),
    pytest.param((1, 1, 1, 1), 4, 3, None, None, None, 1, False, id="all-padding"),
    pytest.param(
        ((2, 2), (2, 2)),
        4,
        3,
        (5, 2),
        (15, 6, 1),
        (8, 2),
        2,
        True,
        id="all-padding-with-frequency",
    ),
    pytest.param(
        (0, 0, 3, 0), 5, 3, None, None, None, 0, True, id="one-non-padding-occurrence"
    ),
    pytest.param(2, 4, 5, None, (2,), (12, 2), -1, True, id="scalar-index"),
    pytest.param(2, 4, 5, None, None, None, 2, True, id="scalar-padding"),
    pytest.param(
        (((0, 2), (1, 0)), ((2, 1), (0, 2))),
        4,
        3,
        None,
        None,
        None,
        1,
        True,
        id="three-dimensional-indices",
    ),
    pytest.param(
        (0, 0, 0), 1, 1, None, None, None, -1, True, id="one-weight-one-feature"
    ),
    pytest.param(
        (0, 5, 0, 5),
        6,
        33,
        None,
        None,
        None,
        -1,
        False,
        id="first-last-rows-and-wide-embedding",
    ),
    pytest.param((0, 3, 3), 4, 1, None, None, None, 3, True, id="last-row-padding"),
    pytest.param(
        (0, 3, 3),
        4,
        1,
        None,
        None,
        None,
        -1,
        True,
        id="minus-one-does-not-mean-last-row-padding",
    ),
    pytest.param((), 0, 3, None, None, None, -1, False, id="empty-vocabulary"),
    pytest.param(
        (), 0, 3, None, None, None, -1, True, id="empty-vocabulary-with-frequency"
    ),
    pytest.param(
        (0, 1, 1), 4, 0, None, None, None, -1, False, id="zero-embedding-dimension"
    ),
    pytest.param(
        (0, 1, 1),
        4,
        0,
        None,
        None,
        None,
        -1,
        True,
        id="zero-embedding-dimension-with-frequency",
    ),
)

_DTYPES = (
    pytest.param(torch.float32, 1e-5, 1e-5, id="float32"),
    pytest.param(torch.float64, 1e-10, 1e-10, id="float64"),
    pytest.param(torch.float16, 1e-2, 1e-2, id="float16"),
    pytest.param(torch.bfloat16, 5e-2, 5e-2, id="bfloat16"),
)


def _embedding_dense_backward(
    grad_output,
    indices,
    grad_weight,
    *,
    num_weights,
    padding_idx,
    scale_grad_by_freq,
    implementation_index,
):
    infini.ops.embedding_dense_backward(
        grad_output,
        indices,
        num_weights,
        padding_idx,
        scale_grad_by_freq,
        grad_weight,
        implementation_index=implementation_index,
        stream=get_stream(grad_output.device),
    )
    return grad_weight


def _torch_embedding_dense_backward(
    grad_output,
    indices,
    grad_weight,
    *,
    num_weights,
    padding_idx,
    scale_grad_by_freq,
):
    # Some Torch CUDA versions fail for empty frequency-scaling inputs or
    # zero feature width. Both have an identically zero/empty dense result.
    if indices.numel() == 0 or grad_weight.numel() == 0:
        grad_weight.zero_()
        return grad_weight
    result = torch.ops.aten.embedding_dense_backward.default(
        grad_output,
        indices.long(),
        num_weights,
        padding_idx,
        scale_grad_by_freq,
    )
    assert result.layout == torch.strided
    grad_weight.copy_(result)
    return grad_weight


def _oracle(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq):
    # Independent mathematical reference: accumulate rows in CPU FP64.
    # This does NOT call embedding_dense_backward, and copies the source storage
    # before any possibly overlapping output can be written.
    index = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    dim = grad_output.shape[-1]
    values = grad_output.detach().to(device="cpu", dtype=torch.float64).clone()
    values = values.reshape(index.numel(), dim)
    mask = index != padding_idx
    index = index[mask]
    values = values[mask]
    result = torch.zeros((num_weights, dim), dtype=torch.float64)
    if index.numel():
        if scale_grad_by_freq:
            counts = torch.bincount(index, minlength=num_weights)
            values = values / counts[index, None]
        result.index_add_(0, index, values)
    return result.to(device=grad_output.device, dtype=grad_output.dtype)


def _check(
    grad_output,
    indices,
    grad_weight,
    *,
    num_weights,
    padding_idx,
    scale_grad_by_freq,
    implementation_index,
    rtol,
    atol,
):
    expected = _oracle(
        grad_output,
        indices,
        num_weights,
        padding_idx,
        scale_grad_by_freq,
    )
    _embedding_dense_backward(
        grad_output,
        indices,
        grad_weight,
        num_weights=num_weights,
        padding_idx=padding_idx,
        scale_grad_by_freq=scale_grad_by_freq,
        implementation_index=implementation_index,
    )
    assert grad_weight.layout == torch.strided
    assert torch.isfinite(grad_weight).all().item()
    torch.testing.assert_close(grad_weight, expected, rtol=rtol, atol=atol)
    return grad_weight


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "indices_values,num_weights,embedding_dim,indices_strides,"
    "grad_output_strides,grad_weight_strides,padding_idx,scale_grad_by_freq",
    _TEST_CASES,
)
@pytest.mark.parametrize("indices_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
def test_embedding_dense_backward(
    indices_values,
    num_weights,
    embedding_dim,
    indices_strides,
    grad_output_strides,
    grad_weight_strides,
    padding_idx,
    scale_grad_by_freq,
    indices_dtype,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    values = torch.tensor(indices_values, dtype=indices_dtype)
    indices = empty_strided(
        values.shape, indices_strides, dtype=indices_dtype, device=device
    )
    indices.copy_(values.to(device))
    grad_output = randn_strided(
        (*indices.shape, embedding_dim),
        grad_output_strides,
        dtype=dtype,
        device=device,
    )
    grad_weight = empty_strided(
        (num_weights, embedding_dim),
        grad_weight_strides,
        dtype=dtype,
        device=device,
    )
    grad_weight.fill_(float("nan"))
    return Payload(
        lambda *args: _embedding_dense_backward(
            *args,
            num_weights=num_weights,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
            implementation_index=implementation_index,
        ),
        lambda *args: _torch_embedding_dense_backward(
            *args,
            num_weights=num_weights,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq,
        ),
        (grad_output, indices, grad_weight),
        {},
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("padding_idx", (-1, 1, 3))
@pytest.mark.parametrize("scale_grad_by_freq", (False, True))
@pytest.mark.parametrize("indices_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
def test_embedding_dense_backward_analytic(
    padding_idx,
    scale_grad_by_freq,
    indices_dtype,
    dtype,
    rtol,
    atol,
    implementation_index,
    device,
):
    indices = torch.tensor([1, 1, 3], dtype=indices_dtype, device=device)
    grad_output = torch.tensor([[2, 4], [8, 12], [3, 9]], dtype=dtype, device=device)
    grad_weight = torch.full((5, 2), float("nan"), dtype=dtype, device=device)
    expected = torch.zeros_like(grad_weight)
    if padding_idx != 1:
        expected[1] = torch.tensor(
            [5, 8] if scale_grad_by_freq else [10, 16],
            dtype=dtype,
            device=device,
        )
    if padding_idx != 3:
        expected[3] = torch.tensor([3, 9], dtype=dtype, device=device)
    _check(
        grad_output,
        indices,
        grad_weight,
        num_weights=5,
        padding_idx=padding_idx,
        scale_grad_by_freq=scale_grad_by_freq,
        implementation_index=implementation_index,
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(grad_weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale_grad_by_freq", (False, True))
@pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
def test_embedding_dense_backward_padding_nan_and_unused_rows(
    scale_grad_by_freq,
    dtype,
    rtol,
    atol,
    implementation_index,
    device,
):
    indices = torch.tensor([1, 1, 2, 4], device=device)
    grad_output = torch.tensor(
        [[float("nan"), float("nan")], [float("nan"), float("nan")], [2, -3], [5, 7]],
        dtype=dtype,
        device=device,
    )
    grad_weight = torch.full((6, 2), float("nan"), dtype=dtype, device=device)
    _check(
        grad_output,
        indices,
        grad_weight,
        num_weights=6,
        padding_idx=1,
        scale_grad_by_freq=scale_grad_by_freq,
        implementation_index=implementation_index,
        rtol=rtol,
        atol=atol,
    )
    # Padding is skipped entirely, including its NaN upstream gradient.
    assert torch.count_nonzero(grad_weight[[0, 1, 3, 5]]).item() == 0


@pytest.mark.parametrize("all_padding", (False, True))
@pytest.mark.parametrize("scale_grad_by_freq", (False, True))
@pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
def test_embedding_dense_backward_empty_or_all_padding_zeroes_output(
    all_padding,
    scale_grad_by_freq,
    dtype,
    rtol,
    atol,
    implementation_index,
    device,
):
    indices = torch.full(
        (2, 3) if all_padding else (2, 0), 1, dtype=torch.int64, device=device
    )
    grad_output = torch.full(
        (*indices.shape, 3), float("nan"), dtype=dtype, device=device
    )
    grad_weight = torch.full((4, 3), float("nan"), dtype=dtype, device=device)
    _check(
        grad_output,
        indices,
        grad_weight,
        num_weights=4,
        padding_idx=1,
        scale_grad_by_freq=scale_grad_by_freq,
        implementation_index=implementation_index,
        rtol=rtol,
        atol=atol,
    )
    assert torch.count_nonzero(grad_weight).item() == 0


@pytest.mark.parametrize("scale_grad_by_freq", (False, True))
def test_embedding_dense_backward_transposed_layout(
    scale_grad_by_freq,
    implementation_index,
    device,
):
    indices = torch.tensor([[0, 1], [1, 2], [1, 3]], device=device).t()
    grad_output = torch.randn((3, 2, 5), device=device).transpose(0, 1)
    grad_weight = torch.full((5, 4), float("nan"), device=device).t()
    _check(
        grad_output,
        indices,
        grad_weight,
        num_weights=4,
        padding_idx=2,
        scale_grad_by_freq=scale_grad_by_freq,
        implementation_index=implementation_index,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("pattern", ("zero", "cancel", "negative"))
@pytest.mark.parametrize("scale_grad_by_freq", (False, True))
@pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
def test_embedding_dense_backward_gradient_values(
    pattern,
    scale_grad_by_freq,
    dtype,
    rtol,
    atol,
    implementation_index,
    device,
):
    indices = torch.tensor([1, 1, 3], device=device)
    values = {
        "zero": [[0, 0], [0, 0], [0, 0]],
        "cancel": [[2, -4], [-2, 4], [0, 0]],
        "negative": [[-2, -4], [-8, -12], [-3, -9]],
    }[pattern]
    grad_output = torch.tensor(values, dtype=dtype, device=device)
    grad_weight = torch.full((5, 2), float("nan"), dtype=dtype, device=device)
    _check(
        grad_output,
        indices,
        grad_weight,
        num_weights=5,
        padding_idx=-1,
        scale_grad_by_freq=scale_grad_by_freq,
        implementation_index=implementation_index,
        rtol=rtol,
        atol=atol,
    )
    if pattern in ("zero", "cancel"):
        assert torch.count_nonzero(grad_weight).item() == 0


@pytest.mark.parametrize("alias", ("full", "partial", "strided"))
@pytest.mark.parametrize("scale_grad_by_freq", (False, True))
@pytest.mark.parametrize("dtype,rtol,atol", _DTYPES)
def test_embedding_dense_backward_grad_output_alias(
    alias,
    scale_grad_by_freq,
    dtype,
    rtol,
    atol,
    implementation_index,
    device,
):
    indices = torch.tensor([2, 0, 2], device=device)
    if alias == "full":
        grad_output = torch.arange(1, 7, dtype=dtype, device=device).reshape(3, 2)
        grad_weight = grad_output
    elif alias == "partial":
        backing = torch.arange(1, 13, dtype=dtype, device=device)
        grad_output = backing[:6].view(3, 2)
        grad_weight = backing[3:9].view(3, 2)
    else:
        backing = torch.arange(1, 13, dtype=dtype, device=device).reshape(3, 4)
        grad_output = backing[:, :2]
        grad_weight = backing[:, 1:3]
    _check(
        grad_output,
        indices,
        grad_weight,
        num_weights=3,
        padding_idx=-1,
        scale_grad_by_freq=scale_grad_by_freq,
        implementation_index=implementation_index,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("partial", (False, True))
@pytest.mark.parametrize(
    "indices_dtype,dtype",
    (
        (torch.int32, torch.float32),
        (torch.int64, torch.float64),
    ),
)
def test_embedding_dense_backward_indices_alias(
    partial,
    indices_dtype,
    dtype,
    implementation_index,
    device,
):
    # Equal element sizes allow index and output views over the same bytes.
    backing = torch.zeros((12,), dtype=dtype, device=device)
    indices = backing[1:7].view(indices_dtype)
    indices.copy_(torch.tensor([2, 0, 2, 1, 0, 2], dtype=indices_dtype, device=device))
    grad_weight = backing[4:10].view(3, 2) if partial else backing[1:7].view(3, 2)
    grad_output = torch.arange(1, 13, dtype=dtype, device=device).view(6, 2)
    _check(
        grad_output,
        indices,
        grad_weight,
        num_weights=3,
        padding_idx=-1,
        scale_grad_by_freq=True,
        implementation_index=implementation_index,
        rtol=1e-10 if dtype == torch.float64 else 1e-5,
        atol=1e-10 if dtype == torch.float64 else 1e-5,
    )


def test_embedding_dense_backward_strided_output_guards(implementation_index, device):
    indices = torch.tensor([1, 1, 3], device=device)
    grad_output = torch.tensor(
        [[2, 4], [8, 12], [3, 9]],
        dtype=torch.float32,
        device=device,
    )
    backing = torch.full((5, 5), 17.0, device=device)
    grad_weight = backing[:, 1::2]
    expected_backing = backing.clone()
    expected_backing[:, 1::2] = _oracle(grad_output, indices, 5, -1, True)
    _check(
        grad_output,
        indices,
        grad_weight,
        num_weights=5,
        padding_idx=-1,
        scale_grad_by_freq=True,
        implementation_index=implementation_index,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(backing, expected_backing)


@pytest.mark.parametrize("scale_grad_by_freq", (False, True))
def test_embedding_dense_backward_overwrites_previous_result(
    scale_grad_by_freq,
    implementation_index,
    device,
):
    indices = torch.tensor([1, 1, 3], device=device)
    grad_output = torch.ones((3, 2), device=device)
    grad_weight = torch.full((5, 2), 17.0, device=device)
    for values in ([1, 1, 3], [0, 2, 2]):
        indices.copy_(torch.tensor(values, device=device))
        _check(
            grad_output,
            indices,
            grad_weight,
            num_weights=5,
            padding_idx=-1,
            scale_grad_by_freq=scale_grad_by_freq,
            implementation_index=implementation_index,
            rtol=1e-5,
            atol=1e-5,
        )
