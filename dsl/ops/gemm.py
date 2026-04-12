from dsl.decorators import manual_op


@manual_op(
    name="Gemm",
    base="src/base/gemm.h",
    impl_names={0: "cublas", 1: "cublaslt"},
    backends={
        "cuda": {"include": "cuda/gemm/blas.h", "class": "BlasGemm", "blas": True},
        "ascend": "ascend/gemm/kernel.h",
        "cambricon": "cambricon/gemm/cnblas.h",
        "cpu": "cpu/gemm/gemm.h",
    },
)
def gemm():
    ...
