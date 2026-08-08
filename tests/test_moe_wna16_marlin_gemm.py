import infini.ops
import pytest
import torch

from tests.utils import get_stream


if not hasattr(infini.ops, "MoeWna16MarlinGemm"):
    pytest.skip(
        "`MoeWna16MarlinGemm` is not available on this platform",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "dtype, b_q_type_id, num_bits, rtol, atol",
    (
        (torch.float16, 1125899907892224, 4, 2e-2, 2e-2),
        (torch.bfloat16, 1125899907892224, 4, 5e-2, 5e-2),
        (torch.float16, 1125899923621888, 8, 2e-2, 2e-2),
    ),
)
def test_moe_wna16_marlin_gemm(
    dtype,
    b_q_type_id,
    num_bits,
    rtol,
    atol,
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("`moe_wna16_marlin_gemm` requires the NVIDIA backend")

    provider_case = _make_case(device, dtype, b_q_type_id, num_bits)
    case = _make_case(device, dtype, b_q_type_id, num_bits)
    expected = provider_case["out"]
    provider_result = _call_provider(provider_case, expected)

    assert provider_result.data_ptr() == expected.data_ptr()
    result = _call_infini(case, implementation_index, get_stream(case["a"].device))

    assert result is None
    torch.testing.assert_close(case["out"], expected, rtol=rtol, atol=atol)


def test_moe_wna16_marlin_gemm_non_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    provider_case = _make_case(device, torch.float16, 1125899907892224, 4)
    case = _make_case(device, torch.float16, 1125899907892224, 4)
    expected = provider_case["out"]
    _call_provider(provider_case, expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    _call_infini(case, implementation_index, stream.cuda_stream)

    stream.synchronize()
    torch.testing.assert_close(case["out"], expected, rtol=2e-2, atol=2e-2)


def test_moe_wna16_marlin_gemm_optional_act_order(device, implementation_index):
    if device != "cuda":
        pytest.skip("activation-order metadata requires the NVIDIA backend")

    provider_case = _make_case(
        device, torch.float16, 1125899907892224, 4, has_act_order=True
    )
    case = _make_case(device, torch.float16, 1125899907892224, 4, has_act_order=True)
    expected = provider_case["out"]
    _call_provider(provider_case, expected)

    _call_infini(case, implementation_index, get_stream(case["a"].device))

    torch.testing.assert_close(case["out"], expected, rtol=2e-2, atol=2e-2)


def test_moe_wna16_marlin_gemm_zero_points(device, implementation_index):
    if device != "cuda":
        pytest.skip("zero-point metadata requires the NVIDIA backend")

    provider_case = _make_case(
        device, torch.float16, 1125899906843648, 4, has_zero_points=True
    )
    case = _make_case(device, torch.float16, 1125899906843648, 4, has_zero_points=True)
    expected = provider_case["out"]
    provider_result = _call_provider(provider_case, expected)

    assert provider_result.data_ptr() == expected.data_ptr()
    _call_infini(case, implementation_index, get_stream(case["a"].device))

    torch.testing.assert_close(case["out"], expected, rtol=2e-2, atol=2e-2)


def _make_case(
    device,
    dtype,
    b_q_type_id,
    num_bits,
    *,
    has_act_order=False,
    has_zero_points=False,
):
    torch.manual_seed(0)
    size_m, size_n, size_k = 1, 128, 256
    top_k = 2
    num_experts = 4
    moe_block_size = 16
    route_count = size_m * top_k
    padding = route_count
    sorted_token_ids = torch.tensor(
        (0,) + (padding,) * 15 + (1,) + (padding,) * 45,
        dtype=torch.int32,
        device=device,
    )
    pack_factor = 32 // num_bits
    num_groups = 8 if has_act_order else 1
    scale = 0.0 if has_act_order else 0.02
    g_idx_or_none = None
    perm_or_none = None
    if has_act_order:
        g_idx_or_none = (
            torch.arange(num_groups, dtype=torch.int32, device=device)
            .repeat_interleave(size_k // num_groups)
            .repeat(num_experts, 1)
        )
        perm_or_none = torch.arange(size_k, dtype=torch.int32, device=device).repeat(
            num_experts, 1
        )

    b_zeros_or_none = None
    if has_zero_points:
        b_zeros_or_none = torch.zeros(
            (num_experts, num_groups, size_n // pack_factor),
            dtype=torch.int32,
            device=device,
        )

    return {
        "a": torch.randn((size_m, size_k), dtype=dtype, device=device),
        "b_q_weight": torch.zeros(
            (num_experts, size_k // 16, size_n * 16 // pack_factor),
            dtype=torch.int32,
            device=device,
        ),
        "b_scales": torch.full(
            (num_experts, num_groups, size_n), scale, dtype=dtype, device=device
        ),
        "global_scale": None,
        "b_zeros_or_none": b_zeros_or_none,
        "g_idx_or_none": g_idx_or_none,
        "perm_or_none": perm_or_none,
        "workspace": torch.zeros(432, dtype=torch.int32, device=device),
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": torch.tensor((0, 1, 0, 0), dtype=torch.int32, device=device),
        "num_tokens_past_padded": torch.tensor((32,), dtype=torch.int32, device=device),
        "topk_weights": torch.tensor(
            ((0.75, 0.25),), dtype=torch.float32, device=device
        ),
        "moe_block_size": moe_block_size,
        "top_k": top_k,
        "mul_topk_weights": True,
        "is_ep": False,
        "b_q_type_id": b_q_type_id,
        "size_m": size_m,
        "size_n": size_n,
        "size_k": size_k,
        "is_full_k": True,
        "use_atomic_add": dtype == torch.float16,
        "use_fp32_reduce": True,
        "is_zp_float": False,
        "out": torch.zeros((route_count, size_n), dtype=dtype, device=device),
    }


def _call_provider(case, out):
    return torch.ops._moe_C.moe_wna16_marlin_gemm(
        case["a"],
        out,
        case["b_q_weight"],
        case["b_scales"],
        case["global_scale"],
        case["b_zeros_or_none"],
        case["g_idx_or_none"],
        case["perm_or_none"],
        case["workspace"],
        case["sorted_token_ids"],
        case["expert_ids"],
        case["num_tokens_past_padded"],
        case["topk_weights"],
        case["moe_block_size"],
        case["top_k"],
        case["mul_topk_weights"],
        case["is_ep"],
        case["b_q_type_id"],
        case["size_m"],
        case["size_n"],
        case["size_k"],
        case["is_full_k"],
        case["use_atomic_add"],
        case["use_fp32_reduce"],
        case["is_zp_float"],
    )


def _call_infini(case, implementation_index, stream):
    return infini.ops.moe_wna16_marlin_gemm(
        case["a"],
        case["b_q_weight"],
        case["b_scales"],
        case["global_scale"],
        case["b_zeros_or_none"],
        case["g_idx_or_none"],
        case["perm_or_none"],
        case["workspace"],
        case["sorted_token_ids"],
        case["expert_ids"],
        case["num_tokens_past_padded"],
        case["topk_weights"],
        case["moe_block_size"],
        case["top_k"],
        case["mul_topk_weights"],
        case["is_ep"],
        case["b_q_type_id"],
        case["size_m"],
        case["size_n"],
        case["size_k"],
        case["is_full_k"],
        case["use_atomic_add"],
        case["use_fp32_reduce"],
        case["is_zp_float"],
        case["out"],
        stream=stream,
        implementation_index=implementation_index,
    )
