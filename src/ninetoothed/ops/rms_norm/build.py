import functools
import itertools

import ninetoothed
import ntops

_BLOCK_SIZES = (256, 512)

_DTYPES = ("float32", "float16", "bfloat16")

_DEFAULT_NDIMS = (2, 3, 4)

_BATCH_DIM_AUTO_TUNE_SIZE = 1
_NORMALIZED_DIM_AUTO_TUNE_SIZE = 256

_CONFIGS = tuple(
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
        (getattr(ninetoothed, name) for name in _DTYPES),
        _BLOCK_SIZES,
    )
)


def _shape_options(ndim, num_normalized_dims):
    batch_dims = ndim - num_normalized_dims

    return (
        *({"upper_bound": _BATCH_DIM_AUTO_TUNE_SIZE} for _ in range(batch_dims)),
        *(
            {"upper_bound": _NORMALIZED_DIM_AUTO_TUNE_SIZE}
            for _ in range(num_normalized_dims)
        ),
    )


def _premake(
    ndim,
    num_normalized_dims,
    input_dtype=None,
    weight_dtype=None,
    output_dtype=None,
    block_size=None,
):
    dims = tuple(-(dim + 1) for dim in range(num_normalized_dims))
    arrangement = functools.partial(
        ntops.kernels.reduction.arrangement,
        dim=dims,
        block_size=block_size,
    )
    shape_options = _shape_options(ndim, num_normalized_dims)
    tensors = (
        ninetoothed.Tensor(
            ndim,
            other=0,
            dtype=input_dtype,
            shape_options=shape_options,
        ),
        ninetoothed.Tensor(ndim, dtype=weight_dtype, shape_options=shape_options),
        ninetoothed.Tensor(0, dtype=ninetoothed.float32),
        ninetoothed.Tensor(ndim, dtype=output_dtype, shape_options=shape_options),
        ninetoothed.Tensor(0, dtype=ninetoothed.uint64),
    )

    return arrangement, ntops.kernels.rms_norm.application, tensors


def build(output_dir):
    variant_dir = output_dir / "rms_norm"
    variant_dir.mkdir(parents=True, exist_ok=True)
    ninetoothed.build(
        _premake,
        _CONFIGS,
        meta_parameters=("block_size",),
        caller="cuda",
        kernel_name="infini_ops_ninetoothed_rms_norm",
        output_dir=variant_dir,
        lazy=False,
    )
