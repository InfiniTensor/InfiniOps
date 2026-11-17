import ninetoothed
import ninetoothed.language as ntl

_CONFIGS = (((), {}, {}),)


def premake():
    tile_shape = (1, 1, 128)

    def arrangement(
        qkv,
        qkv_output,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        num_heads_q,
        eps,
        rotary_dim,
        is_neox,
    ):
        return (
            qkv.tile(tile_shape),
            qkv_output.tile(tile_shape),
            q_weight.tile(tile_shape),
            k_weight.tile(tile_shape),
            positions.tile(tile_shape),
            cos_sin_cache,
            num_heads_q,
            eps,
            rotary_dim,
            is_neox,
        )

    def application(
        qkv,
        qkv_output,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        num_heads_q,
        eps,
        rotary_dim,
        is_neox,
    ):
        input_f32 = ntl.cast(qkv, ntl.float32)
        scale = ntl.rsqrt(ntl.sum(input_f32 * input_f32) / qkv.source.shape[-1] + eps)

        token_index = ntl.cast(qkv.offsets(0)[:, None, None], ntl.int64)
        head_index = ntl.cast(qkv.offsets(1)[:, None, None], ntl.int64)
        dimension = ntl.cast(qkv.offsets(2), ntl.int64)
        weight = ntl.where(
            head_index < num_heads_q,
            ntl.cast(q_weight, ntl.float32),
            ntl.cast(k_weight, ntl.float32),
        )

        half_rotary = ntl.cast(rotary_dim // 2, ntl.int64)
        first_in_pair = dimension < half_rotary
        pair_distance = half_rotary
        pair_dimension = ntl.cast(
            dimension + ntl.where(first_in_pair, pair_distance, -pair_distance),
            ntl.int64,
        )
        frequency_dimension = ntl.cast(dimension % half_rotary, ntl.int64)
        q_pair_weight = ntl.cast(q_weight.source[0, 0, pair_dimension], ntl.float32)
        k_pair_weight = ntl.cast(k_weight.source[0, 0, pair_dimension], ntl.float32)
        pair_weight = ntl.where(
            head_index < num_heads_q,
            q_pair_weight,
            k_pair_weight,
        )

        position = ntl.cast(positions, ntl.int64)
        cos = ntl.cast(cos_sin_cache.source[position, frequency_dimension], ntl.float32)
        sin = ntl.cast(
            cos_sin_cache.source[position, half_rotary + frequency_dimension],
            ntl.float32,
        )
        paired_input = ntl.cast(
            qkv.source[token_index, head_index, pair_dimension], ntl.float32
        )

        normalized = input_f32 * weight * scale
        paired_normalized = paired_input * pair_weight * scale
        rotation_sign = ntl.where(first_in_pair, -1.0, 1.0)
        output = normalized * cos + rotation_sign * paired_normalized * sin
        qkv_output = ntl.cast(output, qkv_output.dtype)  # noqa: F841

    return (
        arrangement,
        application,
        (
            ninetoothed.Tensor(3, other=0, dtype="bfloat16"),
            ninetoothed.Tensor(3, other=0, dtype="bfloat16"),
            ninetoothed.Tensor(3, dtype="bfloat16"),
            ninetoothed.Tensor(3, dtype="bfloat16"),
            ninetoothed.Tensor(3, dtype=ninetoothed.int64),
            ninetoothed.Tensor(2, dtype="bfloat16"),
            ninetoothed.Tensor(0, dtype=ninetoothed.int64),
            ninetoothed.Tensor(0, dtype=ninetoothed.float64),
            ninetoothed.Tensor(0, dtype=ninetoothed.int64),
            ninetoothed.Tensor(0, dtype=ninetoothed.int64),
        ),
    )


def build(output_dir):
    variant_dir = output_dir / "fused_qk_norm_rope"
    variant_dir.mkdir(parents=True, exist_ok=True)
    ninetoothed.build(
        premake,
        _CONFIGS,
        caller="cuda",
        kernel_name="infini_ops_ninetoothed_fused_qk_norm_rope",
        output_dir=variant_dir,
        lazy=False,
    )
