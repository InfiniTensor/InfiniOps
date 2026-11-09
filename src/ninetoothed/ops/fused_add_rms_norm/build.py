import itertools

import ninetoothed
import ninetoothed.language as ntl

_BLOCK_SIZE = 1024

_DTYPES = ("float32", "float16", "bfloat16")

_DEFAULT_NDIMS = (2, 3, 4)

_CONFIGS = tuple(
    (
        (),
        {
            "ndim": ndim,
            "dtype": dtype,
            "block_size": _BLOCK_SIZE,
        },
        {},
    )
    for ndim, dtype in itertools.product(
        _DEFAULT_NDIMS,
        (getattr(ninetoothed, name) for name in _DTYPES),
    )
)


def premake(ndim, dtype=None, block_size=None):
    if block_size is None:
        raise ValueError("FusedAddRmsNorm requires an explicit block size")

    tile_shape = (1,) * (ndim - 1) + (block_size,)

    def arrangement_(input, residual, weight, has_weight, eps, num_normalized_elements):
        def arrange(tensor):
            return tensor.tile(tile_shape)

        return (
            arrange(input),
            arrange(residual),
            arrange(weight),
            has_weight,
            eps,
            num_normalized_elements,
        )

    def application(input, residual, weight, has_weight, eps, num_normalized_elements):
        input_f32 = ntl.cast(input, ntl.float32)
        residual_f32 = ntl.cast(residual, ntl.float32)
        summed = input_f32 + residual_f32
        rounded_sum = ntl.cast(summed, residual.dtype)
        normalized = ntl.cast(rounded_sum, ntl.float32)
        scale = ntl.rsqrt(
            ntl.sum(normalized * normalized) / num_normalized_elements + eps
        )
        residual = rounded_sum
        weight_multiplier = ntl.cast(has_weight, ntl.float32)
        weight_multiplier = weight_multiplier * ntl.cast(weight, ntl.float32) + (
            1 - weight_multiplier
        )
        input = ntl.cast(normalized * scale * weight_multiplier, input.dtype)

    return (
        arrangement_,
        application,
        (
            ninetoothed.Tensor(ndim, other=0, dtype=dtype),
            ninetoothed.Tensor(ndim, other=0, dtype=dtype),
            ninetoothed.Tensor(ndim, dtype=dtype),
            ninetoothed.Tensor(0, dtype=ninetoothed.int32),
            ninetoothed.Tensor(0, dtype=ninetoothed.float64),
            ninetoothed.Tensor(0, dtype=ninetoothed.int64),
        ),
    )


def build(output_dir):
    variant_dir = output_dir / "fused_add_rms_norm"
    variant_dir.mkdir(parents=True, exist_ok=True)
    ninetoothed.build(
        premake,
        _CONFIGS,
        meta_parameters=("block_size",),
        caller="cuda",
        kernel_name="infini_ops_ninetoothed_fused_add_rms_norm",
        output_dir=variant_dir,
        lazy=False,
    )
