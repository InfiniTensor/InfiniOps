"""DSL alternative implementation for RmsNorm (impl_index=1).

Registers as ``Operator<RmsNorm, kDev, 1>`` alongside the existing
hand-written ``Operator<RmsNorm, kDev, 0>``.
"""

from dsl.decorators import infini_op
from dsl.primitives import Scalar, Tensor, reduce_mean, rsqrt


@infini_op(
    name="RmsNorm",
    impl_index=1,
    shapes={"B": "batch_size", "H": "nhead", "D": "dim"},
    manual_backends={
        "ascend": "ascend/rms_norm/kernel.h",
        "cambricon": "cambricon/rms_norm/rms_norm.h",
    },
)
def rms_norm_dsl(
    input: Tensor["B", "H", "D"],
    weight: Tensor["D"],
    eps: Scalar[float] = 1e-6,
) -> Tensor["B", "H", "D"]:
    ss = reduce_mean(input * input, dim="D")
    rms = rsqrt(ss + eps)

    return input * rms * weight
