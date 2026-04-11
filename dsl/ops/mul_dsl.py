"""DSL alternative implementation for Mul (impl_index=1).

Registers as ``Operator<Mul, kDev, 1>`` alongside the existing
hand-written ``Operator<Mul, kDev, 0>``.
"""

from dsl.decorators import infini_op
from dsl.primitives import Tensor


@infini_op(
    name="Mul",
    impl_index=1,
    shapes={"N": "output_size"},
    manual_backends={
        "ascend": "ascend/mul/kernel.h",
    },
)
def mul_dsl(
    input: Tensor["N"],
    other: Tensor["N"],
) -> Tensor["N"]:
    return input * other
