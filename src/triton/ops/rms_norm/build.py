from scripts.triton import aot

_DTYPES = ("fp16", "bf16", "fp32")
_BLOCK_SIZES = (2048,)
_ALIGNMENTS = (16, None)
_NUM_WARPS = 8
_NUM_STAGES = 3
_DATA_PTRS = ("x_ptr", "w_ptr", "y_ptr")
_I32_SCALARS = ("m", "n")
_I64_SCALARS = (
    "stride_xm",
    "stride_xn",
    "stride_wn",
    "stride_ym",
    "stride_yn",
)


def _signature(dtype, block_size, alignment):
    return aot.Signature(
        pointer_dtypes={name: dtype for name in _DATA_PTRS},
        pointer_alignments={name: alignment for name in _DATA_PTRS},
        scalar_dtypes={
            "eps": "fp32",
            **{name: "i32" for name in _I32_SCALARS},
            **{name: "i64" for name in _I64_SCALARS},
        },
        constexprs={"BLOCK_SIZE": block_size},
    )


def configs():
    for dtype in _DTYPES:
        yield tuple(
            aot.CompileConfig(
                signature=_signature(dtype, block_size, alignment),
                grid="m, 1, 1",
                out_name=f"infini_ops_triton_rms_norm_{dtype}",
                num_warps=_NUM_WARPS,
                num_stages=_NUM_STAGES,
            )
            for block_size in _BLOCK_SIZES
            for alignment in _ALIGNMENTS
        )
