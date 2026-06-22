from scripts.triton import aot

_DTYPES = ("fp16", "bf16", "fp32")
_BLOCK_SIZES = ((64, 64, 32), (128, 64, 32), (64, 128, 32))
_ALIGNMENTS = (16, None)
_NUM_WARPS = 4
_NUM_STAGES = 3
_GROUP_SIZE_M = 8
_DATA_PTRS = ("a_ptr", "b_ptr", "c_ptr")
_I32_SCALARS = ("m", "n", "k", "batch_count")
_I64_SCALARS = (
    "stride_am",
    "stride_ak",
    "stride_bk",
    "stride_bn",
    "stride_cm",
    "stride_cn",
    "batch_stride_a",
    "batch_stride_b",
    "batch_stride_c",
)


def _signature(dtype, block_size, alignment):
    block_m, block_n, block_k = block_size
    return aot.Signature(
        pointer_dtypes={name: dtype for name in _DATA_PTRS},
        pointer_alignments={name: alignment for name in _DATA_PTRS},
        scalar_dtypes={
            "alpha": "fp64",
            "beta": "fp64",
            **{name: "i32" for name in _I32_SCALARS},
            **{name: "i64" for name in _I64_SCALARS},
        },
        constexprs={
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": _GROUP_SIZE_M,
        },
    )


def configs():
    for dtype in _DTYPES:
        yield tuple(
            aot.CompileConfig(
                signature=_signature(dtype, block_size, alignment),
                grid=(
                    f"((m + {block_size[0]} - 1) / {block_size[0]}) * "
                    f"((n + {block_size[1]} - 1) / {block_size[1]}), batch_count, 1"
                ),
                out_name=f"infini_ops_triton_gemm_{dtype}",
                num_warps=_NUM_WARPS,
                num_stages=_NUM_STAGES,
            )
            for block_size in _BLOCK_SIZES
            for alignment in _ALIGNMENTS
        )
