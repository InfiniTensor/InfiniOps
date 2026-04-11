"""DSL alternative implementation for Add (impl_index=1).

Registers as ``Operator<Add, kDev, 1>`` alongside the existing
hand-written ``Operator<Add, kDev, 0>``.
"""

from dsl.decorators import infini_op
from dsl.primitives import Tensor


@infini_op(
    name="Add",
    impl_index=1,
    shapes={"N": "output_size"},
    manual_backends={
        "ascend": "ascend/add/kernel.h",
    },
)
def add_dsl(
    input: Tensor["N"],
    other: Tensor["N"],
) -> Tensor["N"]:
    return input + other
