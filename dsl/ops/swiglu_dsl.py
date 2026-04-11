"""DSL alternative implementation for Swiglu (impl_index=1).

SwiGLU(input, gate) = input * silu(gate).

Registers as ``Operator<Swiglu, kDev, 1>`` alongside the existing
hand-written ``Operator<Swiglu, kDev, 0>``.
"""

from dsl.decorators import infini_op
from dsl.primitives import Tensor, silu


@infini_op(
    name="Swiglu",
    impl_index=1,
    shapes={"N": "output_size"},
    manual_backends={
        "ascend": "ascend/swiglu/kernel.h",
    },
)
def swiglu_dsl(
    input: Tensor["N"],
    other: Tensor["N"],
) -> Tensor["N"]:
    return input * silu(other)
