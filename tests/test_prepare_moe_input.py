import subprocess
import sys
import textwrap

import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize(
    "topk_ids_values, num_experts, n, k",
    (
        (((2, 0), (1, 2), (0, 3), (3, 1)), 4, 64, 128),
        (((4, 1), (0, 4), (2, 1), (4, 2), (0, 1)), 6, 96, 256),
    ),
)
def test_prepare_moe_input(
    topk_ids_values,
    num_experts,
    n,
    k,
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("`prepare_moe_input` requires the NVIDIA backend")

    topk_ids = torch.tensor(topk_ids_values, dtype=torch.int32, device=device)
    outputs = _make_outputs(topk_ids, num_experts)

    _prepare_moe_input(
        topk_ids,
        num_experts,
        n,
        k,
        *outputs,
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, num_experts, n, k, *outputs)


def test_prepare_moe_input_blockscale_offsets(device, implementation_index):
    if device != "cuda":
        pytest.skip("`prepare_moe_input` requires the NVIDIA backend")

    topk_ids = torch.cat(
        (
            torch.zeros(130, dtype=torch.int32),
            torch.ones(129, dtype=torch.int32),
            torch.full((1,), 2, dtype=torch.int32),
        )
    ).reshape(130, 2)
    topk_ids = topk_ids.to(device)
    num_experts = 4
    n = 80
    k = 192
    outputs = _make_outputs(topk_ids, num_experts, with_blockscale=True)

    _prepare_moe_input(
        topk_ids,
        num_experts,
        n,
        k,
        *outputs,
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, num_experts, n, k, *outputs)


def test_prepare_moe_input_empty_tokens(device, implementation_index):
    if device != "cuda":
        pytest.skip("`prepare_moe_input` requires the NVIDIA backend")

    topk_ids = torch.empty((0, 2), dtype=torch.int32, device=device)
    num_experts = 4
    n = 64
    k = 128
    outputs = _make_outputs(topk_ids, num_experts, with_blockscale=True)

    _prepare_moe_input(
        topk_ids,
        num_experts,
        n,
        k,
        *outputs,
        implementation_index=implementation_index,
    )

    _assert_matches_reference(topk_ids, num_experts, n, k, *outputs)


def test_prepare_moe_input_descriptor_reuses_matching_metadata(device):
    if device != "cuda":
        pytest.skip("`prepare_moe_input` requires the NVIDIA backend")

    num_experts = 4
    n = 64
    k = 128
    topk_ids = torch.tensor(
        ((0, 1), (2, 3), (3, 0), (1, 2)), dtype=torch.int32, device=device
    )
    outputs = _make_outputs(topk_ids, num_experts, with_blockscale=True)
    operator = infini.ops.PrepareMoeInput(
        topk_ids,
        num_experts,
        n,
        k,
        *outputs,
    )
    reused_topk_ids = torch.tensor(
        ((3, 2), (1, 0), (0, 2), (3, 1)), dtype=torch.int32, device=device
    )
    reused_outputs = _make_outputs(reused_topk_ids, num_experts, with_blockscale=True)

    result = operator(
        reused_topk_ids,
        num_experts,
        n,
        k,
        *reused_outputs,
    )

    assert result is None
    _assert_matches_reference(reused_topk_ids, num_experts, n, k, *reused_outputs)


@pytest.mark.parametrize(
    "metadata_change",
    (
        "topk_strides",
        "topk_dtype",
        "problem_sizes_shape",
        "blockscale_presence",
        "num_experts",
    ),
)
def test_prepare_moe_input_descriptor_rejects_changed_metadata(metadata_change, device):
    if device != "cuda":
        pytest.skip("`prepare_moe_input` requires the NVIDIA backend")

    assertion_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            _DESCRIPTOR_REUSE_SCRIPT,
            "assertions_enabled_probe",
        ],
        capture_output=True,
        text=True,
    )

    if assertion_probe.returncode == 0:
        pytest.skip("descriptor validation requires an assertions-enabled build")

    result = subprocess.run(
        [sys.executable, "-c", _DESCRIPTOR_REUSE_SCRIPT, metadata_change],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    expected_message = (
        "`PrepareMoeInput` attributes changed after descriptor creation"
        if metadata_change == "num_experts"
        else "`PrepareMoeInput` tensor metadata differs from its descriptor"
    )
    assert expected_message in result.stderr


def test_prepare_moe_input_non_default_stream(device, implementation_index):
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

    infini.ops.prepare_moe_input(
        topk_ids,
        num_experts,
        n,
        k,
        *outputs,
        stream=stream.cuda_stream,
        implementation_index=implementation_index,
    )

    stream.synchronize()
    _assert_matches_reference(topk_ids, num_experts, n, k, *outputs)


def test_prepare_moe_input_device_guard():
    if not infini.ops.PrepareMoeInput.active_implementation_indices("nvidia"):
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

        infini.ops.prepare_moe_input(
            topk_ids,
            num_experts,
            n,
            k,
            *outputs,
            stream=stream.cuda_stream,
        )

        assert torch.cuda.current_device() == 0
        stream.synchronize()
        _assert_matches_reference(topk_ids, num_experts, n, k, *outputs)
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
    input_permutation = torch.empty(numel, dtype=torch.int32, device=topk_ids.device)
    output_permutation = torch.empty_like(input_permutation)
    outputs = (
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
    )

    if with_blockscale:
        blockscale_offsets = torch.empty_like(expert_offsets)
        outputs += (blockscale_offsets,)

    return outputs


def _prepare_moe_input(
    topk_ids,
    num_experts,
    n,
    k,
    *outputs,
    implementation_index,
):
    infini.ops.prepare_moe_input(
        topk_ids,
        num_experts,
        n,
        k,
        *outputs,
        stream=get_stream(topk_ids.device),
        implementation_index=implementation_index,
    )


def _assert_matches_reference(
    topk_ids,
    num_experts,
    n,
    k,
    expert_offsets,
    problem_sizes1,
    problem_sizes2,
    input_permutation,
    output_permutation,
    blockscale_offsets=None,
):
    flattened = topk_ids.flatten().to(torch.int64)
    counts = torch.bincount(flattened, minlength=num_experts)
    expected_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device=counts.device), counts.cumsum(0))
    )
    expected_problem_sizes1 = torch.stack(
        (
            counts,
            torch.full_like(counts, 2 * n),
            torch.full_like(counts, k),
        ),
        dim=1,
    )
    expected_problem_sizes2 = torch.stack(
        (
            counts,
            torch.full_like(counts, k),
            torch.full_like(counts, n),
        ),
        dim=1,
    )

    torch.testing.assert_close(expert_offsets, expected_offsets.to(torch.int32))
    torch.testing.assert_close(problem_sizes1, expected_problem_sizes1.to(torch.int32))
    torch.testing.assert_close(problem_sizes2, expected_problem_sizes2.to(torch.int32))

    output_permutation_values = output_permutation.to("cpu").tolist()
    input_permutation_values = input_permutation.to("cpu").tolist()
    expert_offsets_values = expected_offsets.to("cpu").tolist()
    flattened_values = flattened.to("cpu").tolist()
    topk = topk_ids.size(1)

    assert sorted(output_permutation_values) == list(range(topk_ids.numel()))

    for token_expert_index, expert_id in enumerate(flattened_values):
        expert_major_index = output_permutation_values[token_expert_index]
        assert (
            expert_offsets_values[expert_id]
            <= expert_major_index
            < expert_offsets_values[expert_id + 1]
        )
        assert input_permutation_values[expert_major_index] == (
            token_expert_index // topk
        )

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


_DESCRIPTOR_REUSE_SCRIPT = textwrap.dedent(
    r"""
    import sys

    import infini.ops
    import torch


    metadata_change = sys.argv[1]
    num_experts = 4
    n = 64
    k = 128
    topk_ids = torch.tensor(((0, 1), (2, 3)), dtype=torch.int32, device="cuda")
    expert_offsets = torch.empty(5, dtype=torch.int32, device="cuda")
    problem_sizes1 = torch.empty((4, 3), dtype=torch.int32, device="cuda")
    problem_sizes2 = torch.empty_like(problem_sizes1)
    input_permutation = torch.empty(4, dtype=torch.int32, device="cuda")
    output_permutation = torch.empty_like(input_permutation)
    blockscale_offsets = torch.empty_like(expert_offsets)
    operator = infini.ops.PrepareMoeInput(
        topk_ids,
        num_experts,
        n,
        k,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        blockscale_offsets,
    )

    if metadata_change in ("assertions_enabled_probe", "num_experts"):
        num_experts += 1
    elif metadata_change == "topk_strides":
        topk_ids = topk_ids.T
    elif metadata_change == "topk_dtype":
        topk_ids = topk_ids.to(torch.int64)
    elif metadata_change == "problem_sizes_shape":
        problem_sizes1 = torch.empty((4, 4), dtype=torch.int32, device="cuda")

    args = (
        topk_ids,
        num_experts,
        n,
        k,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
    )
    args += (
        None if metadata_change == "blockscale_presence" else blockscale_offsets,
    )

    operator(*args)
    torch.cuda.synchronize()
    """
)
