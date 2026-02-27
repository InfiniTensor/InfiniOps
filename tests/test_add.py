import infini.ops
import pytest
import torch

from tests.utils import empty_strided, get_available_devices, randn_strided


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-3, 1e-3),
    ),
)
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

    output = empty_strided(shape, c_strides, dtype=dtype, device=device)
    expected = output.clone()

    # TODO: Add keyword argument support.
    infini.ops.add(a, b, output)
    torch.add(a, b, out=expected)

    assert torch.allclose(output, expected, rtol=rtol, atol=atol)
