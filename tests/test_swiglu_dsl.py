"""Tests for the DSL-generated Swiglu operator (implementation_index=1).

Validates that the DSL-generated code produces results identical to
the reference: SwiGLU(input, gate) = input * silu(gate).
"""

import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, rand_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, gate_strides, out_strides",
    (
        ((13, 4), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
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
def test_swiglu_dsl(
    shape, input_strides, gate_strides, out_strides, dtype, device, rtol, atol
):
    input = rand_strided(shape, input_strides, dtype=dtype, device=device)
    gate = rand_strided(shape, gate_strides, dtype=dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        _swiglu_dsl, _torch_swiglu, (input, gate, out), {}, rtol=rtol, atol=atol
    )


def _swiglu_dsl(input, gate, out):
    infini.ops.swiglu(input, gate, out, implementation="dsl")

    return out


def _torch_swiglu(input, gate, out):
    swish_x = gate * torch.sigmoid(gate)

    return torch.mul(input, swish_x, out=out)
