from dsl.decorators import manual_op


@manual_op(
    name="Linear",
    base="src/base/linear.h",
    backends={
        "cuda": "cuda/linear/kernel.h",
        "ascend": "ascend/linear/kernel.h",
        "cpu": "cpu/linear/linear.h",
    },
)
def linear():
    ...
