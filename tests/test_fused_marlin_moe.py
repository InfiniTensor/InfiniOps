import importlib

import infini.ops
import pytest
import torch

from tests.utils import get_stream


if not hasattr(infini.ops, "FusedMarlinMoe"):
    pytest.skip(
        "`FusedMarlinMoe` is not available on this platform",
        allow_module_level=True,
    )
if 16 not in infini.ops.FusedMarlinMoe.active_implementation_indices("nvidia"):
    pytest.skip(
        "the linked vLLM `FusedMarlinMoe` implementation is unavailable",
        allow_module_level=True,
    )


try:
    importlib.import_module("vllm.model_executor.layers.fused_moe.fused_marlin_moe")
except (ImportError, OSError, RuntimeError) as error:
    pytest.skip(
        f"vLLM `fused_marlin_moe` reference is unavailable: {error}",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "inplace, has_expert_map, has_empty_act_order",
    (
        (False, False, False),
        (True, False, False),
        (False, True, True),
    ),
)
@pytest.mark.parametrize("device, implementation_index", (("cuda", 16),))
def test_fused_marlin_moe(
    inplace,
    has_expert_map,
    has_empty_act_order,
    device,
    implementation_index,
):
    provider_case = _make_case(device, inplace, has_expert_map, has_empty_act_order)
    case = _make_case(device, inplace, has_expert_map, has_empty_act_order)
    expected = _call_provider(provider_case)

    assert (expected.data_ptr() == provider_case["hidden_states"].data_ptr()) is inplace
    result = _call_infini(
        case,
        implementation_index,
        get_stream(case["hidden_states"].device),
    )

    assert result is None
    torch.testing.assert_close(case["out"], expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("device, implementation_index", (("cuda", 16),))
def test_fused_marlin_moe_non_default_stream(device, implementation_index):
    provider_case = _make_case(device, False, False)
    case = _make_case(device, False, False)
    expected = _call_provider(provider_case)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    _call_infini(case, implementation_index, stream.cuda_stream)

    stream.synchronize()
    torch.testing.assert_close(case["out"], expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "unsupported, message",
    (
        ("float4", "float4_e2m1f"),
        ("global_scale", "global_scale"),
        ("short_workspace", "workspace"),
    ),
)
@pytest.mark.parametrize("device, implementation_index", (("cuda", 16),))
def test_fused_marlin_moe_unsupported(
    unsupported,
    message,
    device,
    implementation_index,
):
    case = _make_case(device, False, False)
    if unsupported == "float4":
        case["quant_type_id"] = 562949953487106
    elif unsupported == "global_scale":
        case["global_scale1"] = torch.ones(1, device=device)
    else:
        case["workspace"] = torch.zeros(1, dtype=torch.int32, device=device)

    with pytest.raises(RuntimeError, match=message):
        _call_infini(
            case,
            implementation_index,
            get_stream(case["hidden_states"].device),
        )


def _make_case(device, inplace, has_expert_map, has_empty_act_order=False):
    torch.manual_seed(0)
    num_tokens, hidden_size = 1, 256
    intermediate_size = 128
    num_experts = 4
    pack_factor = 8
    hidden_states = torch.randn(
        (num_tokens, hidden_size), dtype=torch.float16, device=device
    )

    def make_packed_weight(shape):
        return torch.randint(
            -(2**31),
            2**31 - 1,
            shape,
            dtype=torch.int32,
            device=device,
        )

    w1 = make_packed_weight(
        (
            num_experts,
            hidden_size // 16,
            intermediate_size * 2 * 16 // pack_factor,
        )
    )
    w2 = make_packed_weight(
        (
            num_experts,
            intermediate_size // 16,
            hidden_size * 16 // pack_factor,
        )
    )
    w1_scale = (
        torch.rand(
            (num_experts, 1, intermediate_size * 2),
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    ).to(hidden_states.dtype)
    w2_scale = (
        torch.rand(
            (num_experts, 1, hidden_size),
            dtype=torch.float32,
            device=device,
        )
        * 0.02
    ).to(hidden_states.dtype)
    gating_output = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device=device
    )
    topk_weights = torch.tensor(((0.75, 0.25),), dtype=torch.float32, device=device)
    topk_ids = torch.tensor(((0, 1),), dtype=torch.int32, device=device)
    expert_map = (
        torch.arange(num_experts, dtype=torch.int32, device=device)
        if has_expert_map
        else None
    )
    act_order = (
        tuple(torch.empty(0, dtype=torch.int32, device=device) for _ in range(4))
        if has_empty_act_order
        else (None,) * 4
    )
    out = hidden_states if inplace else torch.empty_like(hidden_states)

    return {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "gating_output": gating_output,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "quant_type_id": 1125899907892224,
        "apply_router_weight_on_input": False,
        "global_num_experts": num_experts if has_expert_map else -1,
        "expert_map": expert_map,
        "global_scale1": None,
        "global_scale2": None,
        "g_idx1": act_order[0],
        "g_idx2": act_order[1],
        "sort_indices1": act_order[2],
        "sort_indices2": act_order[3],
        "w1_zeros": None,
        "w2_zeros": None,
        "workspace": None,
        "is_k_full": True,
        "inplace": inplace,
        "out": out,
    }


def _call_provider(case):
    return torch.ops.vllm.fused_marlin_moe.default(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["w1_scale"],
        case["w2_scale"],
        case["gating_output"],
        case["topk_weights"],
        case["topk_ids"],
        case["quant_type_id"],
        case["apply_router_weight_on_input"],
        case["global_num_experts"],
        case["expert_map"],
        case["global_scale1"],
        case["global_scale2"],
        case["g_idx1"],
        case["g_idx2"],
        case["sort_indices1"],
        case["sort_indices2"],
        case["w1_zeros"],
        case["w2_zeros"],
        case["workspace"],
        case["is_k_full"],
        case["inplace"],
    )


def _call_infini(case, implementation_index, stream):
    return infini.ops.fused_marlin_moe(
        case["hidden_states"],
        case["w1"],
        case["w2"],
        case["w1_scale"],
        case["w2_scale"],
        case["gating_output"],
        case["topk_weights"],
        case["topk_ids"],
        case["quant_type_id"],
        case["apply_router_weight_on_input"],
        case["global_num_experts"],
        case["expert_map"],
        case["global_scale1"],
        case["global_scale2"],
        case["g_idx1"],
        case["g_idx2"],
        case["sort_indices1"],
        case["sort_indices2"],
        case["w1_zeros"],
        case["w2_zeros"],
        case["workspace"],
        case["is_k_full"],
        case["inplace"],
        case["out"],
        stream=stream,
        implementation_index=implementation_index,
    )
