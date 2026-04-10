from dsl.decorators import manual_op


@manual_op(
    name="Matmul",
    base="src/base/matmul.h",
    backends={
        "ascend": "ascend/matmul/kernel.h",
    },
)
def matmul():
    ...
