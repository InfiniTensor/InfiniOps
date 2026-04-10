"""Example `@infini_op` definition for Add (DSL version).

This demonstrates how a simple binary elementwise operator can be defined
purely in the DSL.  The existing `add.py` stays as `@manual_op` until
migration is complete.
"""

from dsl.decorators import infini_op
from dsl.primitives import Scalar, Tensor


@infini_op(
    name="AddDsl",
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
