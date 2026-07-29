# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Quantized layouts are adapted from vLLM's Marlin tests at commit
# `9b9fc4039c25a6e4fe0ae97361b62edd74b8b47e`.

import infini.ops
import pytest
import torch

from tests.utils import get_stream


if not hasattr(infini.ops, "MarlinGemm"):
    pytest.skip(
        "`MarlinGemm` is not available on this platform",
        allow_module_level=True,
    )


_U4_TYPE_ID = 1125899906843648
_U4B8_TYPE_ID = 1125899907892224
_U8B128_TYPE_ID = 1125899923621888
_GROUP_SIZE = 128


@pytest.mark.parametrize(
    (
        "quantization",
        "num_bits",
        "dtype",
        "size_m",
        "size_k",
        "has_bias",
        "is_k_full",
        "use_atomic_add",
        "use_fp32_reduce",
    ),
    (
        pytest.param(
            "gptq",
            4,
            torch.float16,
            16,
            128,
            False,
            True,
            False,
            False,
            id="gptq-w4",
        ),
        pytest.param(
            "awq",
            4,
            torch.bfloat16,
            32,
            128,
            True,
            True,
            False,
            False,
            id="awq-w4",
        ),
        pytest.param(
            "gptq",
            8,
            torch.float16,
            16,
            256,
            False,
            False,
            False,
            False,
            id="gptq-w8-act-order-partial-k",
        ),
        pytest.param(
            "gptq",
            4,
            torch.float16,
            16,
            4096,
            False,
            True,
            False,
            True,
            id="gptq-w4-fp32-reduce-split-k",
        ),
        pytest.param(
            "gptq",
            4,
            torch.float16,
            16,
            4096,
            False,
            True,
            True,
            False,
            id="gptq-w4-atomic-add-split-k",
        ),
    ),
)
def test_marlin_gemm(
    quantization,
    num_bits,
    dtype,
    size_m,
    size_k,
    has_bias,
    is_k_full,
    use_atomic_add,
    use_fp32_reduce,
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("`marlin_gemm` requires the NVIDIA backend")

    args, out, expected = _make_case(
        quantization,
        num_bits,
        dtype,
        size_m,
        size_k,
        has_bias,
        device,
        padded_a=quantization == "gptq" and num_bits == 4 and size_k == 128,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    result = _marlin_gemm(*args, out, implementation_index=implementation_index)

    assert result is None
    _assert_marlin_close(out, expected)
    assert torch.count_nonzero(args[9]).item() == 0


def test_marlin_gemm_non_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    args, out, expected = _make_case(
        "gptq",
        4,
        torch.float16,
        16,
        128,
        False,
        device,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        result = infini.ops.marlin_gemm(
            *args,
            out,
            stream=stream.cuda_stream,
            implementation_index=implementation_index,
        )

    stream.synchronize()
    assert result is None
    _assert_marlin_close(out, expected)
    assert torch.count_nonzero(args[9]).item() == 0


def test_marlin_gemm_empty(device, implementation_index):
    if device != "cuda":
        pytest.skip("`marlin_gemm` requires the NVIDIA backend")

    args, out, _ = _make_case("gptq", 4, torch.float16, 0, 128, False, device)

    result = _marlin_gemm(*args, out, implementation_index=implementation_index)

    assert result is None
    assert out.numel() == 0
    assert torch.count_nonzero(args[9]).item() == 0


def _make_case(
    quantization,
    num_bits,
    dtype,
    size_m,
    size_k,
    has_bias,
    device,
    *,
    padded_a=False,
    is_k_full=True,
    use_atomic_add=False,
    use_fp32_reduce=False,
):
    size_n = 64
    num_groups = size_k // _GROUP_SIZE
    maximum = 1 << num_bits
    stored_zero = maximum // 2

    values = torch.arange(size_k * size_n, dtype=torch.int64, device=device)
    unpacked_weight = ((values * 17 + 3) % maximum).reshape(size_k, size_n)
    scale_groups = torch.arange(num_groups, dtype=torch.float32, device=device)
    scale_columns = torch.arange(size_n, dtype=torch.float32, device=device)
    logical_scales = (
        0.015625
        + scale_groups[:, None] * 0.001953125
        + (scale_columns[None, :] % 17) * 0.00048828125
    ).to(dtype)
    b_scales = _marlin_permute_scales(logical_scales, size_k, size_n, _GROUP_SIZE)
    scale_for_k = logical_scales.float().repeat_interleave(_GROUP_SIZE, dim=0)

    if padded_a:
        a_storage = torch.randn((size_m, size_k + 16), dtype=dtype, device=device)
        a = a_storage[:, 8 : size_k + 8]
    else:
        a = torch.randn((size_m, size_k), dtype=dtype, device=device)

    g_idx = None
    perm = None
    reference_weight = unpacked_weight
    reference_scales = scale_for_k
    reference_zeros = stored_zero
    if quantization == "gptq":
        repack_perm = None
        if num_bits == 8:
            row_perm = (
                torch.arange(size_k, device=device).reshape(2, -1).t().reshape(-1)
            )
            reference_weight = unpacked_weight[row_perm]
            reference_scales = scale_for_k[row_perm]
            unsorted_g_idx = torch.arange(size_k, device=device) // _GROUP_SIZE
            unsorted_g_idx = unsorted_g_idx[row_perm]
            perm = torch.argsort(unsorted_g_idx).to(torch.int32)
            g_idx = unsorted_g_idx[perm.long()].to(torch.int32).contiguous()
            repack_perm = perm

        packed_weight = _gptq_pack(reference_weight, num_bits)
        b_q_weight = _gptq_marlin_repack(
            packed_weight, size_k, size_n, num_bits, repack_perm
        )
        b_zeros = None
        b_type_id = _U4B8_TYPE_ID if num_bits == 4 else _U8B128_TYPE_ID
    else:
        packed_weight = _awq_pack(unpacked_weight, num_bits)
        b_q_weight = _awq_marlin_repack(packed_weight, size_k, size_n, num_bits)
        zero_groups = torch.arange(num_groups, dtype=torch.int64, device=device)
        zero_columns = torch.arange(size_n, dtype=torch.int64, device=device)
        logical_zeros = (
            zero_groups[:, None] * 3 + zero_columns[None, :] * 5 + 1
        ) % maximum
        b_zeros = _marlin_zero_points(logical_zeros, num_groups, size_n, num_bits)
        reference_zeros = logical_zeros.repeat_interleave(_GROUP_SIZE, dim=0)
        b_type_id = _U4_TYPE_ID

    reference_weight = (reference_weight.float() - reference_zeros) * reference_scales
    expected = torch.matmul(a.float(), reference_weight)

    b_bias = None
    if has_bias:
        bias_columns = torch.arange(size_n, dtype=torch.float32, device=device)
        logical_bias = ((bias_columns % 19) - 9).mul(0.0078125).to(dtype)
        b_bias = _marlin_permute_bias(logical_bias)
        expected = expected + logical_bias.float()

    workspace_size = torch.cuda.get_device_properties(device).multi_processor_count
    workspace = torch.zeros(workspace_size, dtype=torch.int32, device=device)
    out = torch.empty((size_m, size_n), dtype=dtype, device=device)
    args = (
        a,
        b_q_weight,
        b_bias,
        b_scales,
        None,
        None,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_type_id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        False,
    )

    return args, out, expected.to(dtype)


def _gptq_marlin_repack(b_q_weight, size_k, size_n, num_bits, perm=None):
    pack_factor = 32 // num_bits
    if perm is None:
        perm = torch.empty(0, dtype=torch.int32, device=b_q_weight.device)
    out = torch.empty(
        (size_k // 16, size_n * 16 // pack_factor),
        dtype=torch.int32,
        device=b_q_weight.device,
    )
    infini.ops.gptq_marlin_repack(
        b_q_weight,
        perm,
        size_k,
        size_n,
        num_bits,
        False,
        out,
        stream=get_stream(b_q_weight.device),
    )

    return out


def _awq_marlin_repack(b_q_weight, size_k, size_n, num_bits):
    pack_factor = 32 // num_bits
    out = torch.empty(
        (size_k // 16, size_n * 16 // pack_factor),
        dtype=torch.int32,
        device=b_q_weight.device,
    )
    infini.ops.awq_marlin_repack(
        b_q_weight,
        size_k,
        size_n,
        num_bits,
        False,
        out,
        stream=get_stream(b_q_weight.device),
    )

    return out


def _gptq_pack(unpacked, num_bits):
    pack_factor = 32 // num_bits
    packed = torch.zeros(
        (unpacked.size(0) // pack_factor, unpacked.size(1)),
        dtype=torch.int64,
        device=unpacked.device,
    )
    for index in range(pack_factor):
        packed |= unpacked[index::pack_factor] << (num_bits * index)

    return packed.to(torch.int32)


def _awq_pack(unpacked, num_bits):
    pack_factor = 32 // num_bits
    packed_positions = {
        4: (0, 4, 1, 5, 2, 6, 3, 7),
        8: (0, 2, 1, 3),
    }[num_bits]
    packed = torch.zeros(
        (unpacked.size(0), unpacked.size(1) // pack_factor),
        dtype=torch.int64,
        device=unpacked.device,
    )
    for logical_position, packed_position in enumerate(packed_positions):
        packed |= unpacked[:, logical_position::pack_factor] << (
            num_bits * packed_position
        )

    return packed.to(torch.int32)


def _scale_permutations(device):
    scale = []
    for index in range(8):
        scale.extend(index + 8 * offset for offset in range(8))

    single = []
    for index in range(4):
        single.extend(2 * index + offset for offset in (0, 1, 8, 9, 16, 17, 24, 25))

    return (
        torch.tensor(scale, dtype=torch.long, device=device),
        torch.tensor(single, dtype=torch.long, device=device),
    )


def _marlin_permute_scales(scales, size_k, size_n, group_size):
    scale_perm, scale_perm_single = _scale_permutations(scales.device)
    permutation = (
        scale_perm if group_size < size_k and group_size != -1 else scale_perm_single
    )

    return (
        scales.reshape(-1, permutation.numel())[:, permutation]
        .reshape(-1, size_n)
        .contiguous()
    )


def _marlin_permute_bias(bias):
    original_shape = bias.shape
    _, permutation = _scale_permutations(bias.device)

    return (
        bias.reshape(-1, permutation.numel())[:, permutation]
        .reshape(original_shape)
        .contiguous()
    )


def _marlin_zero_points(zero_points, num_groups, size_n, num_bits):
    scale_permutation, _ = _scale_permutations(zero_points.device)
    zero_points = zero_points.reshape(-1, scale_permutation.numel())[
        :, scale_permutation
    ]
    interleave = {
        4: (0, 2, 4, 6, 1, 3, 5, 7),
        8: (0, 2, 1, 3),
    }[num_bits]
    interleave = torch.tensor(interleave, dtype=torch.long, device=zero_points.device)
    zero_points = (
        zero_points.reshape(-1, interleave.numel())[:, interleave]
        .reshape(num_groups, size_n)
        .contiguous()
    )

    return _pack_cols(zero_points, num_bits)


def _pack_cols(values, num_bits):
    pack_factor = 32 // num_bits
    packed = torch.zeros(
        (values.size(0), values.size(1) // pack_factor),
        dtype=torch.int64,
        device=values.device,
    )
    for index in range(pack_factor):
        packed |= values[:, index::pack_factor] << (num_bits * index)

    return packed.to(torch.int32).contiguous()


def _marlin_gemm(*args, implementation_index):
    infini.ops.marlin_gemm(
        *args,
        stream=get_stream(args[0].device),
        implementation_index=implementation_index,
    )


def _assert_marlin_close(actual, expected):
    denominator = expected.float().abs().mean().clamp_min(1e-6)
    relative_error = (actual.float() - expected.float()).abs().mean() / denominator

    assert relative_error.item() < 0.04
