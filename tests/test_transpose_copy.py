import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

_TEST_CASES = (
    pytest.param((2, 3), None, 0, 1, None, id="two-dimensional"),
    pytest.param(
        (2, 3, 4),
        None,
        0,
        2,
        None,
        id="three-dimensional-outer-dimensions",
    ),
    pytest.param((2, 3, 4), None, -1, -2, None, id="negative-dimensions"),
    pytest.param((2, 3, 4), None, 1, 1, None, id="same-dimension"),
    pytest.param(
        (2, 3, 4),
        (20, 6, 1),
        0,
        2,
        (10, 3, 1),
        id="strided-input-and-output",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "input_shape, input_strides, dim0, dim1, out_strides",
    _TEST_CASES,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    tuple((dtype, 0, 0) for dtype in _DTYPES),
)
def test_transpose_copy(
    input_shape,
    input_strides,
    dim0,
    dim1,
    out_strides,
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
    out = empty_strided(
        _out_shape(input_shape, dim0, dim1),
        out_strides,
        dtype=dtype,
        device=device,
    )
    out.fill_(17)

    return Payload(
        lambda *args: _transpose_copy(
            *args,
            implementation_index=implementation_index,
        ),
        _torch_transpose_copy,
        (input, dim0, dim1, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _transpose_copy(input, dim0, dim1, out, *, implementation_index):
    infini.ops.transpose_copy(
        input,
        dim0,
        dim1,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )
    return out


def _torch_transpose_copy(input, dim0, dim1, out):
    result = input.transpose_copy(dim0, dim1).clone()
    out.copy_(result)
    return out


def _out_shape(shape, dim0, dim1):
    result = list(shape)
    dim0 %= len(shape)
    dim1 %= len(shape)
    result[dim0], result[dim1] = result[dim1], result[dim0]
    return tuple(result)


def _span(shape, strides):
    return 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))


def _offsets(placement, input_span, output_span):
    if placement == "same-start":
        return 8, 8
    if placement == "partial-output-after":
        return 8, 10
    if placement == "partial-output-before":
        return 10, 8
    if placement == "adjacent-output-after":
        return 8, 8 + input_span
    if placement == "adjacent-output-before":
        return 8 + output_span, 8
    if placement == "separated-output-after":
        return 8, 8 + input_span + 5
    if placement == "separated-output-before":
        return 8 + output_span + 5, 8
    raise AssertionError(placement)


def _run_alias_case(
    input_shape,
    input_strides,
    dim0,
    dim1,
    out_strides,
    input_offset,
    output_offset,
    *,
    dtype,
    device,
    implementation_index,
    repeats=1,
    special_values=None,
):

    backing = torch.arange(256, dtype=torch.float32, device=device).to(dtype)
    input = backing.as_strided(input_shape, input_strides, input_offset)
    out = backing.as_strided(
        _out_shape(input_shape, dim0, dim1),
        out_strides,
        output_offset,
    )
    if special_values is not None:
        input.copy_(
            torch.tensor(special_values, dtype=dtype, device=device).reshape(
                input_shape
            )
        )

    input_metadata = (tuple(input.shape), input.stride(), input.data_ptr())
    output_metadata = (tuple(out.shape), out.stride(), out.data_ptr())

    for _ in range(repeats):
        expected = input.detach().clone().transpose_copy(dim0, dim1).clone()
        expected_backing = backing.clone()
        expected_backing.as_strided(
            out.shape,
            out.stride(),
            out.storage_offset(),
        ).copy_(expected)

        _transpose_copy(
            input,
            dim0,
            dim1,
            out,
            implementation_index=implementation_index,
        )

        torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(
            backing,
            expected_backing,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        if special_values is not None:
            torch.testing.assert_close(
                torch.signbit(backing),
                torch.signbit(expected_backing),
                rtol=0,
                atol=0,
            )
        assert (tuple(input.shape), input.stride(), input.data_ptr()) == input_metadata
        assert (tuple(out.shape), out.stride(), out.data_ptr()) == output_metadata


_ALIAS_LAYOUTS = (
    pytest.param((3, 3), (3, 1), 0, 1, (3, 1), id="square"),
    pytest.param((3, 3), (3, 1), -1, -2, (3, 1), id="square-negative-dims"),
    pytest.param((3, 3), (1, 3), 0, 1, (3, 1), id="transpose_copyd-input"),
    pytest.param((3, 3), (3, 1), 0, 1, (1, 3), id="transpose_copyd-output"),
    pytest.param((3, 3), (8, 2), 0, 1, (9, 2), id="gapped-input-and-output"),
    pytest.param((2, 3), (3, 1), 0, 1, (2, 1), id="rectangular"),
    pytest.param((2, 3), (10, 2), 0, 1, (7, 2), id="rectangular-gapped"),
    pytest.param(
        (2, 3, 4),
        (30, 7, 1),
        0,
        2,
        (20, 5, 1),
        id="three-dimensional",
    ),
    pytest.param(
        (2, 3, 4),
        (30, 7, 1),
        -1,
        -3,
        (20, 5, 1),
        id="three-dimensional-negative-dims",
    ),
    pytest.param(
        (2, 3),
        (1, 2),
        1,
        1,
        (3, 1),
        id="same-dim-different-layouts",
    ),
    pytest.param(
        (2, 3),
        (8, 2),
        -1,
        1,
        (9, 2),
        id="same-dim-negative-equivalent",
    ),
    pytest.param((1, 4), (9, 2), 0, 1, (2, 11), id="singleton-dimension"),
    pytest.param((7,), (2,), 0, -1, (3,), id="one-dimensional"),
    pytest.param(
        (2, 2, 3, 4),
        (50, 22, 6, 1),
        0,
        3,
        (18, 8, 2, 1),
        id="four-dimensional",
    ),
)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "input_shape, input_strides, dim0, dim1, out_strides",
    _ALIAS_LAYOUTS,
)
@pytest.mark.parametrize(
    "placement",
    (
        "same-start",
        "partial-output-after",
        "partial-output-before",
        "adjacent-output-after",
        "adjacent-output-before",
        "separated-output-after",
        "separated-output-before",
    ),
)
def test_transpose_copy_shared_storage(
    input_shape,
    input_strides,
    dim0,
    dim1,
    out_strides,
    placement,
    dtype,
    device,
    implementation_index,
):
    input_offset, output_offset = _offsets(
        placement,
        _span(input_shape, input_strides),
        _span(_out_shape(input_shape, dim0, dim1), out_strides),
    )
    _run_alias_case(
        input_shape,
        input_strides,
        dim0,
        dim1,
        out_strides,
        input_offset,
        output_offset,
        dtype=dtype,
        device=device,
        implementation_index=implementation_index,
    )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_transpose_copy_review_regression(dtype, device, implementation_index):
    input = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        dtype=dtype,
        device=device,
    )
    out = input.view_as(input)

    _transpose_copy(input, 0, 1, out, implementation_index=implementation_index)

    torch.testing.assert_close(
        out,
        torch.tensor(
            [[1, 4, 7], [2, 5, 8], [3, 6, 9]],
            dtype=dtype,
            device=device,
        ),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "input_shape, input_strides, dim0, dim1, out_strides",
    (
        pytest.param((3, 3), (6, 2), 0, 1, (6, 2), id="square"),
        pytest.param((2, 3), (6, 2), 0, 1, (4, 2), id="rectangular"),
        pytest.param(
            (2, 3, 4),
            (24, 8, 2),
            0,
            2,
            (12, 4, 2),
            id="three-dimensional",
        ),
    ),
)
@pytest.mark.parametrize("reverse", (False, True))
def test_transpose_copy_interleaved_views(
    input_shape,
    input_strides,
    dim0,
    dim1,
    out_strides,
    reverse,
    dtype,
    device,
    implementation_index,
):

    input_offset, output_offset = (9, 8) if reverse else (8, 9)
    _run_alias_case(
        input_shape,
        input_strides,
        dim0,
        dim1,
        out_strides,
        input_offset,
        output_offset,
        dtype=dtype,
        device=device,
        implementation_index=implementation_index,
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("offsets", ((8, 8), (8, 10), (10, 8)))
def test_transpose_copy_repeated_alias_calls(
    offsets,
    dtype,
    device,
    implementation_index,
):
    _run_alias_case(
        (3, 3),
        (3, 1),
        0,
        1,
        (3, 1),
        *offsets,
        dtype=dtype,
        device=device,
        implementation_index=implementation_index,
        repeats=5,
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("offsets", ((8, 8), (8, 10), (10, 8)))
def test_transpose_copy_alias_special_values(
    offsets,
    dtype,
    device,
    implementation_index,
):
    _run_alias_case(
        (3, 3),
        (3, 1),
        -1,
        -2,
        (3, 1),
        *offsets,
        dtype=dtype,
        device=device,
        implementation_index=implementation_index,
        special_values=(
            float("nan"),
            float("inf"),
            -float("inf"),
            0.0,
            -0.0,
            1.0,
            -2.0,
            0.25,
            -0.5,
        ),
    )
