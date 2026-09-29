import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infini.ops
import pytest

import torch
from tests.utils import empty_strided, get_stream


def _ref_moe_align(topk_ids, num_experts, block_size, pad_sorted_token_ids):
    """Reference: pure-Python moe_align (re-implemented here for independence)."""
    numel = topk_ids.numel()

    # Count tokens per expert
    counts = [0] * (num_experts + 1)
    flat_ids = topk_ids.view(-1)
    for i in range(flat_ids.numel()):
        expert_id = int(flat_ids[i].item())
        # No expert_map in fused_dense test path
        expert_id += 1  # Shift by 1: index 0 is reserved
        if 1 <= expert_id <= num_experts:
            counts[expert_id] += 1

    # Padded cumulative sums
    prefix = [0] * (num_experts + 1)
    total_tokens_post_pad = 0
    for i in range(1, num_experts + 1):
        padded = (counts[i] + block_size - 1) // block_size * block_size
        prefix[i] = total_tokens_post_pad + padded
        total_tokens_post_pad = prefix[i]

    # Buffer size must equal the actual padded output length
    max_num_tokens_padded = total_tokens_post_pad
    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    # Fill sorted_token_ids
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,), fill_value=numel, dtype=torch.int32
    )
    positions = [0] * (num_experts + 1)
    for i in range(flat_ids.numel()):
        expert_id = int(flat_ids[i].item())
        expert_id += 1
        if 1 <= expert_id <= num_experts:
            rank = prefix[expert_id - 1] + positions[expert_id]
            sorted_token_ids[rank] = i
            positions[expert_id] += 1

    # expert_ids per block
    expert_ids = torch.full((max_num_blocks,), -1, dtype=torch.int32)
    num_blocks = total_tokens_post_pad // block_size
    for i in range(num_blocks):
        block_start = i * block_size
        expert = 0
        for e in range(1, num_experts + 1):
            if prefix[e] > block_start:
                expert = e - 1
                break
        expert_ids[i] = expert

    num_tokens_post_padded = torch.tensor(
        total_tokens_post_pad, dtype=torch.int32
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def _ref_moe_fused_dense(
    hidden_states, w13, w2, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded
):
    """Reference implementation using PyTorch."""
    num_tokens_post_pad = int(num_tokens_post_padded.item())
    num_tokens = hidden_states.size(0)
    hidden_size = hidden_states.size(1)
    num_experts, twice_inter, _ = w13.shape
    inter_size = twice_inter // 2
    topk = topk_weights.size(1)

    hidden_fp32 = hidden_states.float()
    w13_fp32 = w13.float()
    w2_fp32 = w2.float()

    max_num_tokens_padded = sorted_token_ids.numel()
    max_num_blocks = expert_ids.numel()
    blk_size = (max_num_tokens_padded + max_num_blocks - 1) // max_num_blocks

    numel = num_tokens * topk

    # Gather hidden states into packed buffer.
    # Padding slots (pair_idx == numel) stay zero-filled.
    packed_hidden = torch.zeros(
        num_tokens_post_pad, hidden_size, dtype=torch.float32
    )
    for i in range(num_tokens_post_pad):
        pair_idx = int(sorted_token_ids[i].item())
        if pair_idx < numel:
            token = pair_idx // topk
            packed_hidden[i] = hidden_fp32[token]

    # Determine expert positions from blocks (includes padding slots,
    # because the GPU GEMM runs on the padded count).
    expert_positions = [[] for _ in range(num_experts)]
    if num_tokens_post_pad > 0:
        num_blocks = (num_tokens_post_pad + blk_size - 1) // blk_size
        for b in range(num_blocks):
            expert = int(expert_ids[b].item())
            if expert < 0 or expert >= num_experts:
                continue
            start = b * blk_size
            end = min(start + blk_size, num_tokens_post_pad)
            for pos in range(start, end):
                expert_positions[expert].append(pos)

    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)

    for e in range(num_experts):
        positions = expert_positions[e]
        if len(positions) == 0:
            continue

        expert_input = packed_hidden[positions]  # (m, hidden_size)

        # GEMM1: (m, hidden_size) @ (hidden_size, 2*inter_size)
        gate_up = expert_input @ w13_fp32[e].t()  # (m, 2*inter_size)

        # SwiGLU
        gate = gate_up[:, :inter_size]
        up = gate_up[:, inter_size:]
        silu = gate / (1 + torch.exp(-gate))
        activated = silu * up  # (m, inter_size)

        # GEMM2: (m, inter_size) @ (inter_size, hidden_size)
        out = activated @ w2_fp32[e].t()  # (m, hidden_size)

        # Scatter with weight (skip padding slots, matching the GPU kernel).
        for i_pos, pos in enumerate(positions):
            pair = int(sorted_token_ids[pos].item())
            if pair >= numel:
                continue
            token = pair // topk
            weight = topk_weights.view(-1)[pair].item()
            output[token] += out[i_pos] * weight

    return output


@pytest.mark.parametrize(
    "num_tokens,num_experts,topk,block_size,hidden_size,inter_size",
    (
        (4, 4, 2, 4, 16, 32),
        (8, 8, 2, 4, 16, 32),
        (8, 4, 4, 4, 32, 64),
        (1, 2, 1, 4, 16, 32),
        (16, 8, 2, 8, 32, 64),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 5e-2, 5e-2),
        (torch.bfloat16, 1e-1, 1e-0),
    ),
)
def test_moe_fused_dense(
    num_tokens,
    num_experts,
    topk,
    block_size,
    hidden_size,
    inter_size,
    dtype,
    device,
    rtol,
    atol,
):
    # --- Generate inputs ---
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    gating_output = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)

    # Top-k softmax
    topk_weights = empty_strided(
        (num_tokens, topk), None, dtype=torch.float32, device=device
    )
    topk_ids = empty_strided(
        (num_tokens, topk), None, dtype=torch.int32, device=device
    )
    infini.ops.moe_topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        torch.empty(0, dtype=torch.float32, device=device),
        True,  # renormalize
        0.0,  # moe_softcapping
        stream=get_stream(device),
    )

    # Align -- first run reference on CPU to learn the exact output sizes,
    # then allocate GPU buffers of that size so the C++ layer computes the
    # same block_size that the reference uses.
    ref_sorted_cpu, ref_expert_ids_cpu, _ref_num_post_cpu = _ref_moe_align(
        topk_ids.cpu(), num_experts, block_size, True
    )
    max_num_tokens_padded = ref_sorted_cpu.numel()
    max_num_blocks = ref_expert_ids_cpu.numel()

    sorted_token_ids = empty_strided(
        (max_num_tokens_padded,), None, dtype=torch.int32, device=device
    )
    expert_ids = empty_strided(
        (max_num_blocks,), None, dtype=torch.int32, device=device
    )
    num_tokens_post_padded = empty_strided(
        (1,), None, dtype=torch.int32, device=device
    )
    expert_map = torch.empty(0, dtype=torch.int32, device=device)

    infini.ops.moe_align(
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids,
        expert_map,
        num_experts,
        block_size,
        True,  # pad_sorted_token_ids
        stream=get_stream(device),
    )

    # Weights
    w13 = torch.randn(
        num_experts, 2 * inter_size, hidden_size, dtype=dtype, device=device
    )
    w2 = torch.randn(
        num_experts, hidden_size, inter_size, dtype=dtype, device=device
    )

    # Output
    output = empty_strided(
        (num_tokens, hidden_size), None, dtype=dtype, device=device
    )

    # Call operator
    infini.ops.moe_fused_dense(
        output,
        hidden_states,
        w13,
        w2,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        stream=get_stream(device),
    )

    # Reference
    ref_sorted, ref_expert_ids, ref_num_post = _ref_moe_align(
        topk_ids.cpu(), num_experts, block_size, True
    )
    ref_out = _ref_moe_fused_dense(
        hidden_states.cpu(),
        w13.cpu(),
        w2.cpu(),
        topk_weights.cpu(),
        ref_sorted,
        ref_expert_ids,
        ref_num_post,
    )

    torch.testing.assert_close(output.float().cpu(), ref_out, rtol=rtol, atol=atol)
