"""DSL alternative implementation for Cast (impl_index=1).

Registers as ``Operator<Cast, kDev, 1>`` alongside the existing
hand-written ``Operator<Cast, kDev, 0>``.
"""

from dsl.decorators import infini_op
from dsl.primitives import Tensor, cast


@infini_op(
    name="Cast",
    impl_index=1,
    shapes={"N": "output_size"},
    manual_backends={
        "ascend": "ascend/cast/kernel.h",
    },
)
def cast_dsl(
    input: Tensor["N"],
) -> Tensor["N"]:
    return cast(input)
