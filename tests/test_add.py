import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, randint_strided, randn_strided

_INT_DTYPES = (torch.int16, torch.int32, torch.int64)

_UINT_DTYPES = tuple(
    filter(None, (getattr(torch, f"uint{bits}", None) for bits in (16, 32, 64)))
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, other_strides, out_strides",
    (
        ((13, 4), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((13, 4), (0, 1), None, None),
        ((13, 4, 4), None, None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
        ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
        ((16, 5632), None, None, None),
        ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
        ((13, 16, 2), (128, 4, 1), (0, 2, 1), (64, 4, 1)),
        ((13, 16, 2), (128, 4, 1), (2, 0, 1), (64, 4, 1)),
        ((4, 4, 5632), None, None, None),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
    ),
)
# TODO: Generate implementation indices dynamically from
# `Add.active_implementation_indices` instead of hardcoding.
@pytest.mark.parametrize("implementation_index", (0, 1))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    )
    + tuple((dtype, 0, 0) for dtype in _INT_DTYPES + _UINT_DTYPES),
)
def test_add(
    shape,
    input_strides,
    other_strides,
    out_strides,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    if device == "musa" and dtype in _UINT_DTYPES:
        pytest.skip(
            "The `torch.musa` test cloning path does not support `uint16`, `uint32`, or `uint64`."
        )

    active_indices = infini.ops.Add.active_implementation_indices(device)

    if implementation_index not in active_indices:
        pytest.skip(f"implementation `{implementation_index}` not active on `{device}`")

    if implementation_index == 1 and dtype in _UINT_DTYPES:
        pytest.skip("ATen `add` does not support unsigned integer types")

    if dtype in _INT_DTYPES or dtype in _UINT_DTYPES:
        input = randint_strided(
            0, 100, shape, input_strides, dtype=dtype, device=device
        )
        other = randint_strided(
            0, 100, shape, other_strides, dtype=dtype, device=device
        )
    else:
        input = randn_strided(shape, input_strides, dtype=dtype, device=device)
        other = randn_strided(shape, other_strides, dtype=dtype, device=device)

    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args: _add(*args, implementation_index=implementation_index),
        _torch_add,
        (input, other, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _add(input, other, out, implementation_index=0):
    infini.ops.add(input, other, out, implementation_index=implementation_index)

    return out


def _torch_add(input, other, out):
    if input.dtype in _UINT_DTYPES:
        input = input.to(torch.int64)

    if other.dtype in _UINT_DTYPES:
        other = other.to(torch.int64)

    res = torch.add(input, other)
    out.copy_(res.to(out.dtype))

    return out
