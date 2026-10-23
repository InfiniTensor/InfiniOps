import itertools

import ntops

import ninetoothed

_BLOCK_SIZES = (256, 512)
_DEFAULT_NDIMS = (2, 3, 4)
_DTYPES = {
    "triton": ("float32", "float16", "bfloat16"),
    "ascend": ("float32", "float16"),
}


def _configs(backend):
    return tuple(
        (
            (),
            {
                "ndim": ndim,
                "num_normalized_dims": 1,
                "input_dtype": dtype,
                "weight_dtype": dtype,
                "output_dtype": dtype,
                "block_size": block_size,
            },
            {},
        )
        for ndim, dtype, block_size in itertools.product(
            _DEFAULT_NDIMS,
            (getattr(ninetoothed, name) for name in _DTYPES[backend]),
            _BLOCK_SIZES,
        )
    )


def build(output_dir, *, backend="triton", platform=None):
    if backend not in _DTYPES:
        raise ValueError(f"unsupported NineToothed backend: {backend}")

    caller = "npu" if backend == "ascend" else "cuda"
    platform = platform or ("ascend-910b4" if backend == "ascend" else None)
    variant_dir = output_dir / "rms_norm"
    variant_dir.mkdir(parents=True, exist_ok=True)
    ninetoothed.build(
        ntops.kernels.rms_norm.premake,
        _configs(backend),
        meta_parameters=("block_size",),
        backend=backend,
        platform=platform,
        caller=caller,
        kernel_name="infini_ops_ninetoothed_rms_norm",
        output_dir=variant_dir,
        lazy=False,
    )
