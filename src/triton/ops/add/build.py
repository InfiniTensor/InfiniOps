from scripts.triton import aot

_DTYPES = (
    "fp16",
    "bf16",
    "fp32",
    "fp64",
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
)
_BLOCK_SIZES = (512, 1024)
_ALIGNMENTS = (16, None)
_NUM_WARPS = 4
_NUM_STAGES = 3
_DATA_PTRS = ("x_ptr", "y_ptr", "out_ptr")
_META_PTRS = ("out_shape_ptr", "x_stride_ptr", "y_stride_ptr", "out_stride_ptr")
_SCALARS = ("x_contig", "y_contig", "out_contig", "ndim", "n_elements")


def _signature(dtype, block_size, alignment):
    return aot.Signature(
        pointer_dtypes={
            **{name: dtype for name in _DATA_PTRS},
            **{name: "i64" for name in _META_PTRS},
        },
        pointer_alignments={name: alignment for name in _DATA_PTRS},
        scalar_dtypes={
            **{name: "i32" for name in _SCALARS},
        },
        constexprs={"BLOCK_SIZE": block_size},
    )


def configs():
    for dtype in _DTYPES:
        yield tuple(
            aot.CompileConfig(
                signature=_signature(dtype, block_size, alignment),
                grid=f"(n_elements + {block_size} - 1) / {block_size}, 1, 1",
                out_name=f"infini_ops_triton_add_{dtype}",
                num_warps=_NUM_WARPS,
                num_stages=_NUM_STAGES,
            )
            for block_size in _BLOCK_SIZES
            for alignment in _ALIGNMENTS
        )
