import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infini.ops
import pytest

import torch
from tests.utils import get_stream


def _ref_moe_align(topk_ids, num_experts, block_size, pad_sorted_token_ids):
    """Reference CPU implementation of moe_align."""
    num_tokens, topk = topk_ids.shape
    numel = num_tokens * topk

    # Count tokens per expert
    counts = [0] * (num_experts + 1)
    for i in range(numel):
        expert_id = topk_ids.view(-1)[i].item()
        expert_id += 1  # Shift by 1
        if 1 <= expert_id <= num_experts:
            counts[expert_id] += 1

    # Compute padded cumulative sums
    prefix = [0] * (num_experts + 1)
    total = 0
    for i in range(1, num_experts + 1):
        padded = (counts[i] + block_size - 1) // block_size * block_size
        prefix[i] = total + padded
        total = prefix[i]

    num_tokens_post_padded = total
    max_num_tokens_padded = prefix[num_experts]
    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    # Fill sorted_token_ids
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,), numel, dtype=torch.int32, device=topk_ids.device
    )
    positions = [0] * (num_experts + 1)  # Start from 0, not from counts[expert_id]
    for i in range(numel):
        expert_id = topk_ids.view(-1)[i].item()
        expert_id += 1
        if 1 <= expert_id <= num_experts:
            rank = prefix[expert_id - 1] + positions[expert_id]
            sorted_token_ids[rank] = i
            positions[expert_id] += 1

    # Fill expert_ids
    expert_ids = torch.full(
        (max_num_blocks,), -1, dtype=torch.int32, device=topk_ids.device
    )
    num_blocks = num_tokens_post_padded // block_size
    for i in range(num_blocks):
        block_start = i * block_size
        expert = 0
        for e in range(1, num_experts + 1):
            if prefix[e] > block_start:
                expert = e - 1
                break
        expert_ids[i] = expert

    return sorted_token_ids, expert_ids, torch.tensor(
        [num_tokens_post_padded], dtype=torch.int32, device=topk_ids.device
    )


@pytest.mark.parametrize(
    "num_tokens, topk, num_experts, block_size, pad_sorted_token_ids",
    (
        (4, 2, 2, 4, True),
        (8, 2, 4, 4, True),
        (8, 2, 4, 4, False),
        (16, 2, 8, 8, True),
        (1, 1, 2, 4, True),
    ),
)
def test_moe_align(
    num_tokens,
    topk,
    num_experts,
    block_size,
    pad_sorted_token_ids,
    device,
):
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, topk), dtype=torch.int32, device=device
    )

    ref_sorted, ref_expert_ids, ref_num_tokens = _ref_moe_align(
        topk_ids, num_experts, block_size, pad_sorted_token_ids
    )

    # Kernel skips padding init when pad_sorted_token_ids=False; pre-fill
    # with the sentinel so undefined positions match the reference.
    sorted_token_ids = (
        torch.full_like(ref_sorted, num_tokens * topk)
        if not pad_sorted_token_ids
        else torch.empty_like(ref_sorted)
    )
    expert_ids = torch.empty_like(ref_expert_ids)
    num_tokens_post_padded = torch.empty_like(ref_num_tokens)

    expert_map = torch.empty(0, dtype=torch.int32, device=device)

    infini.ops.moe_align(
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids,
        expert_map,
        num_experts,
        block_size,
        pad_sorted_token_ids,
        stream=get_stream(device),
    )

    torch.testing.assert_close(sorted_token_ids, ref_sorted)
    torch.testing.assert_close(expert_ids, ref_expert_ids)
    torch.testing.assert_close(num_tokens_post_padded, ref_num_tokens)
