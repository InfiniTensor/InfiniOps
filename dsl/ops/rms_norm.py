from dsl.decorators import manual_op


@manual_op(
    name="RmsNorm",
    base="src/base/rms_norm.h",
    backends={
        "cuda": "cuda/rms_norm/kernel.h",
        "ascend": "ascend/rms_norm/kernel.h",
        "cambricon": "cambricon/rms_norm/rms_norm.h",
        "cpu": "cpu/rms_norm/rms_norm.h",
    },
)
def rms_norm():
    ...
