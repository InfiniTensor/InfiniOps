from dsl.decorators import manual_op


@manual_op(
    name="AddRmsNorm",
    base="src/base/add_rms_norm.h",
    backends={
        "ascend": "ascend/add_rms_norm/kernel.h",
    },
)
def add_rms_norm():
    ...
