from dsl.decorators import manual_op


@manual_op(
    name="AddRmsNorm",
    base="src/base/add_rms_norm.h",
    backends={
        "cuda": "cuda/add_rms_norm/kernel.h",
        "ascend": "ascend/add_rms_norm/kernel.h",
    },
)
def add_rms_norm():
    ...
