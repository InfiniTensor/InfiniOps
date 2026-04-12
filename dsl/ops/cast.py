from dsl.decorators import manual_op


@manual_op(
    name="Cast",
    base="src/base/cast.h",
    backends={
        "ascend": "ascend/cast/kernel.h",
        "cpu": "cpu/cast/cast.h",
    },
)
def cast():
    ...
