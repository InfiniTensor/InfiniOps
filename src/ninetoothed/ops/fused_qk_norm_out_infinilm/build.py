import ninetoothed
import ninetoothed.language as ntl

_CONFIGS = (((), {"dtype": ninetoothed.bfloat16}, {}),)


def premake(dtype=None):
    tile_shape = (1, 1, 128)

    def arrangement(
        q0,
        q1,
        k,
        q0_weight,
        q1_weight,
        k_weight,
        eps,
        q0_output,
        q1_output,
        k_output,
        num_normalized_elements,
    ):
        def arrange(tensor):
            return tensor.tile(tile_shape)

        return (
            arrange(q0),
            arrange(q1),
            arrange(k),
            arrange(q0_weight),
            arrange(q1_weight),
            arrange(k_weight),
            eps,
            arrange(q0_output),
            arrange(q1_output),
            arrange(k_output),
            num_normalized_elements,
        )

    def rms(input, weight, eps, num_normalized_elements):
        input_fp32 = ntl.cast(input, ntl.float32)
        scale = ntl.rsqrt(
            ntl.sum(input_fp32 * input_fp32) / num_normalized_elements + eps
        )
        return input_fp32 * weight * scale

    def application(
        q0,
        q1,
        k,
        q0_weight,
        q1_weight,
        k_weight,
        eps,
        q0_output,
        q1_output,
        k_output,
        num_normalized_elements,
    ):
        q0_output = rms(  # noqa: F841
            q0, q0_weight, eps, num_normalized_elements
        )
        q1_output = rms(  # noqa: F841
            q1, q1_weight, eps, num_normalized_elements
        )
        k_output = rms(k, k_weight, eps, num_normalized_elements)  # noqa: F841

    def tensors():
        q0 = ninetoothed.Tensor(3, other=0, dtype=dtype)
        q1 = ninetoothed.Tensor(3, other=0, dtype=dtype)
        k = ninetoothed.Tensor(3, other=0, dtype=dtype)
        q0_weight = ninetoothed.Tensor(3, dtype=dtype)
        q1_weight = ninetoothed.Tensor(3, dtype=dtype)
        k_weight = ninetoothed.Tensor(3, dtype=dtype)
        q0_output = ninetoothed.Tensor(3, dtype=dtype)
        q1_output = ninetoothed.Tensor(3, dtype=dtype)
        k_output = ninetoothed.Tensor(3, dtype=dtype)

        for tensor in (
            q1,
            k,
            q0_weight,
            q1_weight,
            k_weight,
            q0_output,
            q1_output,
            k_output,
        ):
            tensor.shape = q0.shape

        return (
            q0,
            q1,
            k,
            q0_weight,
            q1_weight,
            k_weight,
            ninetoothed.Tensor(0, dtype=ninetoothed.float64),
            q0_output,
            q1_output,
            k_output,
            ninetoothed.Tensor(0, dtype=ninetoothed.int64),
        )

    return arrangement, application, tensors()


def build(output_dir):
    variant_dir = output_dir / "fused_qk_norm_out_infinilm"
    variant_dir.mkdir(parents=True, exist_ok=True)
    ninetoothed.build(
        premake,
        _CONFIGS,
        caller="cuda",
        kernel_name="infini_ops_ninetoothed_fused_qk_norm_out_infinilm",
        output_dir=variant_dir,
        lazy=False,
    )
