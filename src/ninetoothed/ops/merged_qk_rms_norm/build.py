import ninetoothed
import ninetoothed.language as ntl

_CONFIGS = (((), {}, {}),)


def premake():
    tile_shape = (1, 1, 128)

    def arrangement(input, weight, epsilon, output, num_normalized_elements):
        return (
            input.tile(tile_shape),
            weight.tile(tile_shape),
            epsilon,
            output.tile(tile_shape),
            num_normalized_elements,
        )

    def application(input, weight, epsilon, output, num_normalized_elements):
        input_f32 = ntl.cast(input, ntl.float32)
        scale = ntl.rsqrt(
            ntl.sum(input_f32 * input_f32) / num_normalized_elements + epsilon
        )
        output = input_f32 * ntl.cast(weight, ntl.float32) * scale  # noqa: F841

    return (
        arrangement,
        application,
        (
            ninetoothed.Tensor(3, other=0, dtype="bfloat16"),
            ninetoothed.Tensor(3, dtype="bfloat16"),
            ninetoothed.Tensor(0, dtype=ninetoothed.float64),
            ninetoothed.Tensor(3, dtype="bfloat16"),
            ninetoothed.Tensor(0, dtype=ninetoothed.int64),
        ),
    )


def build(output_dir):
    variant_dir = output_dir / "merged_qk_rms_norm"
    variant_dir.mkdir(parents=True, exist_ok=True)
    ninetoothed.build(
        premake,
        _CONFIGS,
        caller="cuda",
        kernel_name="infini_ops_ninetoothed_merged_qk_rms_norm",
        output_dir=variant_dir,
        lazy=False,
    )
