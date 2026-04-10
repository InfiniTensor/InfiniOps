from dsl.decorators import manual_op


@manual_op(
    name="Swiglu",
    base="src/base/swiglu.h",
    backends={
        "cuda": "cuda/swiglu/kernel.h",
        "ascend": "ascend/swiglu/kernel.h",
        "cpu": "cpu/swiglu/swiglu.h",
    },
)
def swiglu():
    ...
