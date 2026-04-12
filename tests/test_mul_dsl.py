"""Tests for the DSL-generated Mul operator (implementation_index=1).

Validates that the DSL-generated CUDA and CPU code produces results
identical to PyTorch's `torch.mul`.
"""

import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, other_strides, out_strides",
    (
        ((13, 4), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((13, 4), (0, 1), None, None),
        ((13, 4, 4), None, None, None),
        ((16, 5632), None, None, None),
        ((4, 4, 5632), None, None, None),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-7, 1e-7),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_mul_dsl(
    shape, input_strides, other_strides, out_strides, dtype, device, rtol, atol
):
    input = randn_strided(shape, input_strides, dtype=dtype, device=device)
    other = randn_strided(shape, other_strides, dtype=dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        _mul_dsl, _torch_mul, (input, other, out), {}, rtol=rtol, atol=atol
    )


def _mul_dsl(input, other, out):
    infini.ops.mul(input, other, out, implementation="dsl")

    return out


def _torch_mul(input, other, out):
    res = torch.mul(input, other)
    out.copy_(res)

    return out
