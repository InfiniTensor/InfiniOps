"""Metax unit tests for the MoE fused-dense operator.

Builds the packed-routing inputs the same way `moe_align_block_size`
produces them (sequential expert buckets, aligned to `block_size`),
then checks the fused gate/up/down projection against a per-expert
torch reference.
"""

import torch
import pytest

import infini.ops

from tests.utils import get_stream


if not hasattr(infini.ops, "moe_fused_dense"):
    pytest.skip(
        "`moe_fused_dense` is not available on this platform",
        allow_module_level=True,
    )


def _make_routing(topk_ids, num_experts, block_size, device):
    """Produce sorted_token_ids/expert_ids/num_tokens_post_pad matching
    the moe_align_block_size sequential-bucket layout."""
    n, topk = topk_ids.shape
    pairs = n * topk
    capacity = pairs + num_experts * (block_size - 1)
    counts = torch.bincount(topk_ids.reshape(-1), minlength=num_experts)
    sorted_token_ids = torch.full((capacity,), pairs, dtype=torch.int32, device=device)
    # Sequential buckets: expert e occupies rows [offset_e, offset_e + count_e).
    row = 0
    token_rows = []
    for e in range(num_experts):
        token_rows.append((e, counts[e].item()))
        row += counts[e].item()
        pad = (-row) % block_size
        row += pad
    flat = topk_ids.reshape(-1)
    row_cursor = 0
    for e in range(num_experts):
        cnt = counts[e].item()
        for i in range(cnt):
            pass
        row_cursor += cnt
        pad = (-row_cursor) % block_size
        row_cursor += pad
    # simpler: compute per-expert rows then scatter
    sorted_token_ids = torch.full((capacity,), pairs, dtype=torch.int32, device=device)
    expert_ids = torch.full(
        ((capacity + block_size - 1) // block_size,),
        num_experts,
        dtype=torch.int32,
        device=device,
    )
    offsets_i = 0
    for e in range(num_experts):
        cnt = int(counts[e])
        block_start = offsets_i
        for j in range(cnt):
            # find (j+1)-th token routed to expert e
            idx = (flat == e).nonzero()[j].item()
            sorted_token_ids[block_start + j] = idx
        offsets_i += cnt
        offsets_i = (offsets_i + block_size - 1) // block_size * block_size
    num_post = torch.tensor([offsets_i], dtype=torch.int32, device=device)
    n_blocks_used = offsets_i // block_size
    expert_ids[:n_blocks_used] = 0
    block_cursor = 0
    for e in range(num_experts):
        cnt = int(counts[e])
        if cnt == 0:
            continue
        start_block = block_cursor
        end_block = (block_cursor * block_size + cnt + block_size - 1) // block_size
        expert_ids[start_block:end_block] = e
        block_cursor = end_block
    return sorted_token_ids, expert_ids, num_post


def _reference(hidden, w13, w2, topk_weights, topk_ids, block_size):
    n, topk = topk_ids.shape
    out = torch.zeros_like(hidden, dtype=torch.float32)
    flat_ids = topk_ids.reshape(-1)
    flat_w = topk_weights.reshape(-1)
    m = w2.shape[2]
    for route in range(n * topk):
        e = int(flat_ids[route])
        token = route // topk
        gate_w, up_w = w13[e][:m].float(), w13[e][m:].float()
        hidden_f = hidden[token].float()
        activated = torch.nn.functional.silu(hidden_f @ gate_w.T) * (hidden_f @ up_w.T)
        out[token] += flat_w[route] * (activated @ w2[e].float().T)
    return out


@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        # The fused-dense implementation computes through the reduced
        # (fp16/bf16) BLAS path; fp32 is rejected at Make time.
        (torch.float16, 2e-2, 2e-2),
        (torch.bfloat16, 4e-2, 4e-2),
    ),
)
def test_moe_fused_dense(dtype, rtol, atol, device):
    if device != "cuda":
        pytest.skip("`moe_fused_dense` numeric test targets the metax/cuda backend")
    torch.manual_seed(0)
    n, h, m, e, topk, block_size = 4, 8, 6, 3, 2, 4
    hidden = torch.randn(n, h, dtype=dtype, device=device).contiguous()
    w13 = torch.randn(e, 2 * m, h, dtype=dtype, device=device).contiguous()
    w2 = torch.randn(e, h, m, dtype=dtype, device=device).contiguous()
    logits = torch.randn(n, e, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, -1), topk, dim=-1)
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()

    sorted_ids, expert_ids, num_post = _make_routing(topk_ids, e, block_size, device)

    output = torch.empty(n, h, dtype=dtype, device=device)
    infini.ops.moe_fused_dense(
        output,
        hidden,
        w13,
        w2,
        topk_weights,
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        block_size,
        stream=get_stream(device),
    )

    expected = _reference(hidden, w13, w2, topk_weights, topk_ids, block_size)
    torch.testing.assert_close(output, expected.to(dtype), rtol=rtol, atol=atol)
