from dsl.decorators import manual_op


@manual_op(
    name="Mul",
    base="src/base/mul.h",
    backends={
        "ascend": "ascend/mul/kernel.h",
        "cpu": "cpu/mul/mul.h",
    },
)
def mul():
    ...
