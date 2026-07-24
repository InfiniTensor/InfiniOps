import infini.ops
import pytest

import torch
from tests.utils import get_stream


def test_get_cutlass_moe_mm_data_binding_exists():
    assert hasattr(infini.ops, "get_cutlass_moe_mm_data")
    assert hasattr(infini.ops, "GetCutlassMoeMmData")


@pytest.mark.parametrize("is_gated", (False, True))
def test_get_cutlass_moe_mm_data_problem_sizes(is_gated, device, implementation_index):
    if device != "cuda":
        pytest.skip("`get_cutlass_moe_mm_data` requires the NVIDIA backend")

    topk_ids = (torch.arange(66, dtype=torch.int32) % 4).reshape(33, 2).to(device)
    num_experts = 4
    n = 64
    k = 128
    outputs = _make_outputs(topk_ids, num_experts)

    _get_cutlass_moe_mm_data(
        topk_ids,
        num_experts,
        n,
        k,
        is_gated,
        *outputs,
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, num_experts, n, k, is_gated, *outputs)


def test_get_cutlass_moe_mm_data_defaults_to_gated_and_swaps_small_problem(
    device, implementation_index
):
    if device != "cuda":
        pytest.skip("`get_cutlass_moe_mm_data` requires the NVIDIA backend")

    topk_ids = torch.tensor(
        ((2, 0), (1, 2), (0, 3), (3, 1)), dtype=torch.int32, device=device
    )
    num_experts = 4
    n = 64
    k = 128
    outputs = _make_outputs(topk_ids, num_experts)

    infini.ops.get_cutlass_moe_mm_data(
        topk_ids,
        num_experts,
        n,
        k,
        *outputs,
        stream=get_stream(topk_ids.device),
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, num_experts, n, k, True, *outputs)


@pytest.mark.parametrize(
    "topk_ids_values",
    (
        ((0, 1), (2, 3), (0, 3), (2, 1)),
        tuple((0, 1) for _ in range(130)),
    ),
)
def test_get_cutlass_moe_mm_data_blockscale_offsets(
    topk_ids_values, device, implementation_index
):
    if device != "cuda":
        pytest.skip("`get_cutlass_moe_mm_data` requires the NVIDIA backend")

    topk_ids = torch.tensor(topk_ids_values, dtype=torch.int32, device=device)
    num_experts = 4
    n = 80
    k = 192
    outputs = _make_outputs(topk_ids, num_experts, with_blockscale=True)

    _get_cutlass_moe_mm_data(
        topk_ids,
        num_experts,
        n,
        k,
        True,
        *outputs,
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, num_experts, n, k, True, *outputs)


def test_get_cutlass_moe_mm_data_invalid_expert_uses_vllm_sentinel(
    device, implementation_index
):
    if device != "cuda":
        pytest.skip("`get_cutlass_moe_mm_data` requires the NVIDIA backend")

    topk_ids = torch.tensor(
        ((0, 1), (2, -1), (3, 0), (-1, 2)), dtype=torch.int32, device=device
    )
    num_experts = 4
    n = 64
    k = 128
    outputs = _make_outputs(topk_ids, num_experts)

    _get_cutlass_moe_mm_data(
        topk_ids,
        num_experts,
        n,
        k,
        True,
        *outputs,
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, num_experts, n, k, True, *outputs)


def test_get_cutlass_moe_mm_data_descriptor_reuses_matching_metadata(device):
    if device != "cuda":
        pytest.skip("`get_cutlass_moe_mm_data` requires the NVIDIA backend")

    num_experts = 4
    n = 64
    k = 128
    topk_ids = torch.tensor(
        ((0, 1), (2, 3), (3, 0), (1, 2)), dtype=torch.int32, device=device
    )
    outputs = _make_outputs(topk_ids, num_experts)
    operator = infini.ops.GetCutlassMoeMmData(
        topk_ids,
        num_experts,
        n,
        k,
        False,
        *outputs,
        None,
    )
    reused_topk_ids = torch.tensor(
        ((3, 2), (1, 0), (0, 2), (3, 1)), dtype=torch.int32, device=device
    )
    reused_outputs = _make_outputs(reused_topk_ids, num_experts)

    result = operator(
        reused_topk_ids,
        num_experts,
        n,
        k,
        False,
        *reused_outputs,
        None,
    )

    assert result is None
    _assert_matches_reference(
        reused_topk_ids, num_experts, n, k, False, *reused_outputs
    )


def test_get_cutlass_moe_mm_data_non_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    topk_ids = torch.tensor(
        ((0, 1), (2, 3), (0, 3), (2, 1)), dtype=torch.int32, device=device
    )
    num_experts = 4
    n = 64
    k = 128
    outputs = _make_outputs(topk_ids, num_experts)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    infini.ops.get_cutlass_moe_mm_data(
        topk_ids,
        num_experts,
        n,
        k,
        True,
        *outputs,
        stream=stream.cuda_stream,
        implementation_index=implementation_index,
    )

    stream.synchronize()
    _assert_matches_reference(topk_ids, num_experts, n, k, True, *outputs)


def test_get_cutlass_moe_mm_data_device_guard():
    if not infini.ops.GetCutlassMoeMmData.active_implementation_indices("nvidia"):
        pytest.skip("device guard test requires the NVIDIA implementation")

    if torch.cuda.device_count() < 2:
        pytest.skip("device guard test requires two NVIDIA GPUs")

    original_device = torch.cuda.current_device()

    try:
        torch.cuda.set_device(0)
        target_device = torch.device("cuda:1")
        topk_ids = torch.tensor(
            ((0, 1), (2, 3), (0, 3), (2, 1)),
            dtype=torch.int32,
            device=target_device,
        )
        num_experts = 4
        n = 64
        k = 128
        outputs = _make_outputs(topk_ids, num_experts)
        stream = torch.cuda.Stream(device=target_device)
        stream.wait_stream(torch.cuda.current_stream(target_device))

        infini.ops.get_cutlass_moe_mm_data(
            topk_ids,
            num_experts,
            n,
            k,
            True,
            *outputs,
            stream=stream.cuda_stream,
        )

        assert torch.cuda.current_device() == 0
        stream.synchronize()
        _assert_matches_reference(topk_ids, num_experts, n, k, True, *outputs)
    finally:
        torch.cuda.set_device(original_device)


def _make_outputs(topk_ids, num_experts, *, with_blockscale=False):
    numel = topk_ids.numel()
    expert_offsets = torch.empty(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )
    problem_sizes1 = torch.empty(
        (num_experts, 3), dtype=torch.int32, device=topk_ids.device
    )
    problem_sizes2 = torch.empty_like(problem_sizes1)
    input_permutation = torch.full(
        (numel,), -1, dtype=torch.int32, device=topk_ids.device
    )
    output_permutation = torch.full_like(input_permutation, -1)
    outputs = (
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
    )

    if with_blockscale:
        outputs += (torch.empty_like(expert_offsets),)

    return outputs


def _get_cutlass_moe_mm_data(
    topk_ids,
    num_experts,
    n,
    k,
    is_gated,
    *outputs,
    implementation_index,
):
    infini.ops.get_cutlass_moe_mm_data(
        topk_ids,
        num_experts,
        n,
        k,
        is_gated,
        *outputs,
        stream=get_stream(topk_ids.device),
        implementation_index=implementation_index,
    )


def _assert_matches_reference(
    topk_ids,
    num_experts,
    n,
    k,
    is_gated,
    expert_offsets,
    problem_sizes1,
    problem_sizes2,
    input_permutation,
    output_permutation,
    blockscale_offsets=None,
):
    flattened = topk_ids.flatten().to(torch.int64)
    valid = (flattened >= 0) & (flattened < num_experts)
    valid_experts = flattened[valid]
    counts = torch.bincount(valid_experts, minlength=num_experts)
    expected_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device=counts.device), counts.cumsum(0))
    )
    swap_ab = blockscale_offsets is None and topk_ids.numel() <= 64
    n1 = 2 * n if is_gated else n
    if swap_ab:
        expected_problem_sizes1 = torch.stack(
            (torch.full_like(counts, n1), counts, torch.full_like(counts, k)),
            dim=1,
        )
        expected_problem_sizes2 = torch.stack(
            (torch.full_like(counts, k), counts, torch.full_like(counts, n)),
            dim=1,
        )
    else:
        expected_problem_sizes1 = torch.stack(
            (counts, torch.full_like(counts, n1), torch.full_like(counts, k)),
            dim=1,
        )
        expected_problem_sizes2 = torch.stack(
            (counts, torch.full_like(counts, k), torch.full_like(counts, n)),
            dim=1,
        )

    torch.testing.assert_close(expert_offsets, expected_offsets.to(torch.int32))
    torch.testing.assert_close(problem_sizes1, expected_problem_sizes1.to(torch.int32))
    torch.testing.assert_close(problem_sizes2, expected_problem_sizes2.to(torch.int32))

    output_values = output_permutation.to("cpu").tolist()
    input_values = input_permutation.to("cpu").tolist()
    offset_values = expected_offsets.to("cpu").tolist()
    flattened_values = flattened.to("cpu").tolist()
    valid_count = int(valid.sum().item())
    topk = topk_ids.size(1)

    assert sorted(
        output_values[index]
        for index, expert in enumerate(flattened_values)
        if 0 <= expert < num_experts
    ) == list(range(valid_count))

    for route, expert in enumerate(flattened_values):
        destination = output_values[route]
        if expert == -1:
            assert destination == valid_count
            continue

        assert offset_values[expert] <= destination < offset_values[expert + 1]
        assert input_values[destination] == route // topk

    if blockscale_offsets is not None:
        rounded_counts = (counts + 127) // 128 * 128
        expected_blockscale_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=counts.device),
                rounded_counts.cumsum(0),
            )
        )
        torch.testing.assert_close(
            blockscale_offsets, expected_blockscale_offsets.to(torch.int32)
        )
