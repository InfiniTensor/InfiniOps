import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, a_strides, b_strides, c_strides",
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
def test_add(shape, a_strides, b_strides, c_strides, dtype, device, rtol, atol):
    a = randn_strided(shape, a_strides, dtype=dtype, device=device)
    b = randn_strided(shape, b_strides, dtype=dtype, device=device)
    c = empty_strided(shape, c_strides, dtype=dtype, device=device)

    return Payload(_add, _torch_add, (a, b, c), {}, rtol=rtol, atol=atol)


def _add(a, b, c):
    infini.ops.add(a, b, c)

    return c


def _torch_add(a, b, c):
    return torch.add(a, b, out=c)
