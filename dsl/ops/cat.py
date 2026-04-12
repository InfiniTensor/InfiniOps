from dsl.decorators import manual_op


@manual_op(
    name="Cat",
    base="src/base/cat.h",
    backends={
        "cuda": "cuda/cat/kernel.h",
        "ascend": "ascend/cat/kernel.h",
        "cpu": "cpu/cat/cat.h",
    },
)
def cat():
    ...
