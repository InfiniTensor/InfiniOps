from dsl.decorators import manual_op


@manual_op(
    name="Gemm",
    base="src/base/gemm.h",
    impl_names={0: "cublas", 1: "cublaslt"},
    backends={
        "cuda": {"include": "cuda/gemm/blas.h", "class": "BlasGemm", "blas": True},
        "nvidia": "nvidia/gemm/cublas.h",
        "metax": "metax/gemm/mcblas.h",
        "iluvatar": "iluvatar/gemm/cublas.h",
        "moore": "moore/gemm/mublas.h",
        "ascend": "ascend/gemm/kernel.h",
        "cambricon": "cambricon/gemm/cnblas.h",
        "cpu": "cpu/gemm/gemm.h",
    },
)
def gemm():
    ...
