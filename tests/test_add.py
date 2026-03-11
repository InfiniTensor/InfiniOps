import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, randint_strided, randn_strided

_INT_DTYPES = (
    torch.int16,
    torch.uint16,
    torch.int32,
    torch.uint32,
    torch.int64,
    torch.uint64,
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
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
        (torch.int16, 0, 0),
        (torch.uint16, 0, 0),
        (torch.int32, 0, 0),
        (torch.uint32, 0, 0),
        (torch.int64, 0, 0),
        (torch.uint64, 0, 0),
    ),
)
def test_add(shape, input_strides, other_strides, out_strides, dtype, device, rtol, atol):
    if dtype in _INT_DTYPES:
        input = randint_strided(0, 100, shape, input_strides, dtype=dtype, device=device)
        other = randint_strided(0, 100, shape, other_strides, dtype=dtype, device=device)
    else:
        input = randn_strided(shape, input_strides, dtype=dtype, device=device)
        other = randn_strided(shape, other_strides, dtype=dtype, device=device)

    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(_add, _torch_add, (input, other, out), {}, rtol=rtol, atol=atol)


def _add(input, other, out):
    infini.ops.add(input, other, out)

    return out


def _torch_add(input, other, out):
    if input.dtype in (torch.uint16, torch.uint32, torch.uint64):
        input = input.to(torch.int64)

    if other.dtype in (torch.uint16, torch.uint32, torch.uint64):
        other = other.to(torch.int64)

    res = torch.add(input, other)
    out.copy_(res.to(out.dtype))

    return out
