from dsl.decorators import manual_op


@manual_op(
    name="Add",
    base="src/base/add.h",
    backends={
        "cuda": "cuda/add/kernel.h",
        "ascend": "ascend/add/kernel.h",
        "cpu": "cpu/add/add.h",
    },
)
def add():
    ...
