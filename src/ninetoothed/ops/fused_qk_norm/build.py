import ninetoothed
import ninetoothed.language as ntl

_CONFIGS = (((), {}, {}),)


def premake():
    tile_shape = (1, 1, 128)

    def arrangement(
        qkv,
        weight,
        eps,
        qkv_output,
        num_normalized_elements,
    ):
        return (
            qkv.tile(tile_shape),
            weight.tile(tile_shape),
            eps,
            qkv_output.tile(tile_shape),
            num_normalized_elements,
        )

    def application(
        qkv,
        weight,
        eps,
        qkv_output,
        num_normalized_elements,
    ):
        input_f32 = ntl.cast(qkv, ntl.float32)
        scale = ntl.rsqrt(
            ntl.sum(input_f32 * input_f32) / num_normalized_elements + eps
        )
        weight = ntl.cast(weight, ntl.float32)
        output = input_f32 * weight * scale
        qkv_output = ntl.cast(output, qkv_output.dtype)  # noqa: F841

    return (
        arrangement,
        application,
        (
            ninetoothed.Tensor(3, other=0, dtype="bfloat16"),
            ninetoothed.Tensor(3, dtype="bfloat16"),
            ninetoothed.Tensor(0, dtype=ninetoothed.float64),
            ninetoothed.Tensor(3, other=0, dtype="bfloat16"),
            ninetoothed.Tensor(0, dtype=ninetoothed.int64),
        ),
    )


def build(output_dir):
    variant_dir = output_dir / "fused_qk_norm"
    variant_dir.mkdir(parents=True, exist_ok=True)
    ninetoothed.build(
        premake,
        _CONFIGS,
        caller="cuda",
        kernel_name="infini_ops_ninetoothed_fused_qk_norm",
        output_dir=variant_dir,
        lazy=False,
    )
