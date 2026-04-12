from dsl.decorators import manual_op


@manual_op(
    name="Matmul",
    base="src/base/matmul.h",
    backends={
        "nvidia": "nvidia/matmul/cublaslt.h",
        "ascend": "ascend/matmul/kernel.h",
        "cpu": "cpu/matmul/matmul.h",
    },
)
def matmul():
    ...
