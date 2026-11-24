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
    if torch.version.hip is not None:
        pytest.skip("the Hygon provider takes block size 16 only; see "
                    "`test_moe_fused_dense_clover`")
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


# ---------------------------------------------------------------------------
# The Triton kernels on every backend that has them, and the native gfx936
# kernels of the Hygon provider, with the routing of Clover: block size 16,
# top-k weights that sum to one, and routes spread over the experts or
# concentrated on a few by the block coreset of the model.

_NATIVE, _TRITON = 0, 10
_CLOVER_BLOCK = 16


def _clover_implementations(device, hidden, intermediate):
    """Returns `(implementation, native)`: the Triton kernels, and on Hygon the
    provider's slot 0, which takes `hidden` and `intermediate` that are
    multiples of 8 of at least 32 with its native kernels and leaves every
    other shape to the Triton kernels."""
    if device != "cuda":
        pytest.skip("requires a CUDA-like backend")
    active = infini.ops.MoeFusedDense.active_implementation_indices(device)
    native = (
        min(hidden, intermediate) >= 32
        and hidden % 8 == 0
        and intermediate % 8 == 0
    )
    pairs = []
    if torch.version.hip is not None and _NATIVE in active:
        if native or _TRITON in active:
            pairs.append((_NATIVE, native))
    if _TRITON in active:
        pairs.append((_TRITON, False))
    if not pairs:
        pytest.skip(f"requires the Triton or the Hygon provider, active={active}")
    return pairs


def _clover_inputs(n, topk, experts, hidden, intermediate, dtype, device, hot, dropped):
    """Inputs of one call. With `hot` > 0 that share of the routes goes to 16
    experts, which gives those experts several blocks; the first `dropped`
    routes of every token are invalidated."""
    scores = -torch.rand(n, experts, device=device)
    if hot > 0:
        crowded = torch.rand(n, experts, device=device) < hot
        scores[:, :16] += 2.0 * crowded[:, :16]
    top = scores.topk(topk, dim=-1)
    weights = top.values.softmax(-1).float().contiguous()
    ids = top.indices.int().contiguous()
    ids[:, :dropped] = -1
    hidden_states = torch.randn(n, hidden, device=device, dtype=dtype) * 0.2
    w13 = (
        torch.randn(experts, 2 * intermediate, hidden, device=device, dtype=dtype)
        * 0.1
    )
    w2 = torch.randn(experts, hidden, intermediate, device=device, dtype=dtype) * 0.1

    pairs = n * topk
    capacity = (
        pairs * _CLOVER_BLOCK
        if pairs < experts
        else pairs + experts * (_CLOVER_BLOCK - 1)
    )
    sorted_ids = torch.empty(capacity, device=device, dtype=torch.int32)
    expert_ids = torch.empty(
        (capacity + _CLOVER_BLOCK - 1) // _CLOVER_BLOCK,
        device=device,
        dtype=torch.int32,
    )
    num_post = torch.empty(1, device=device, dtype=torch.int32)
    infini.ops.moe_align_block_size(
        ids, None, experts, _CLOVER_BLOCK, sorted_ids, expert_ids, num_post,
        stream=get_stream(device),
    )
    return hidden_states, w13, w2, weights, ids, sorted_ids, expert_ids, num_post


def _clover_reference(hidden, w13, w2, weights, ids, round_projections):
    """Float32 throughout, with the activation rounded to the element type
    once. `round_projections` also rounds the gate and up projections, as the
    Triton kernels do. Routes outside `[0, experts)` contribute nothing."""
    dtype, n, intermediate = hidden.dtype, hidden.shape[0], w2.shape[2]
    topk = ids.shape[1]
    out = torch.zeros(n, hidden.shape[1], device=hidden.device, dtype=torch.float32)
    flat = ids.view(-1).long()
    tokens = torch.arange(n, device=hidden.device).repeat_interleave(topk)
    routed = weights.view(-1)
    for expert in flat.unique().tolist():
        if not 0 <= expert < w2.shape[0]:
            continue
        select = (flat == expert).nonzero().squeeze(-1)
        mid = hidden[tokens[select]].float() @ w13[expert].float().T
        if round_projections:
            mid = mid.to(dtype).float()
        gate, up = mid[:, :intermediate], mid[:, intermediate:]
        act = (torch.nn.functional.silu(gate) * up).to(dtype).float()
        out.index_add_(
            0, tokens[select], (act @ w2[expert].float().T) * routed[select, None]
        )
    return out


@pytest.mark.parametrize(
    "n, topk, experts, hidden, intermediate, dtype, hot, dropped",
    (
        # The per-rank shape of Clover-35B, routes spread or concentrated.
        (32, 8, 256, 2048, 256, torch.bfloat16, 0.0, 0),
        (1, 8, 256, 2048, 256, torch.bfloat16, 0.0, 0),
        (64, 8, 256, 2048, 256, torch.bfloat16, 0.8, 0),
        (32, 8, 256, 2048, 256, torch.float16, 0.0, 0),
        (32, 8, 256, 2048, 256, torch.bfloat16, 0.0, 3),
        # Rows and k steps that end inside a tile or a 16-byte unit.
        (20, 4, 16, 136, 40, torch.bfloat16, 0.0, 0),
        (5, 2, 8, 32, 32, torch.float16, 0.0, 0),
        (8, 4, 16, 132, 64, torch.bfloat16, 0.0, 0),
        (32, 8, 64, 192, 24, torch.bfloat16, 0.0, 0),
    ),
)
def test_moe_fused_dense_clover(
    n, topk, experts, hidden, intermediate, dtype, hot, dropped, device
):
    torch.manual_seed(29)
    cases = _clover_implementations(device, hidden, intermediate)
    tensors = _clover_inputs(
        n, topk, experts, hidden, intermediate, dtype, device, hot, dropped
    )
    hidden_states, w13, w2, weights, ids = tensors[:5]

    for implementation, native in cases:
        # The native kernels keep the gate and up projections in float32; the
        # Triton kernels also round them to the element type.
        expected = _clover_reference(
            hidden_states, w13, w2, weights, ids, round_projections=not native
        )
        output = torch.full_like(hidden_states, float("nan"))
        infini.ops.moe_fused_dense(
            output, *tensors, _CLOVER_BLOCK,
            stream=get_stream(device), implementation_index=implementation,
        )
        torch.cuda.synchronize()
        atol = 3e-3 if dtype == torch.bfloat16 else 1e-3
        torch.testing.assert_close(
            output.float(), expected, rtol=2.5e-2, atol=atol,
            msg=lambda m: f"implementation {implementation}: {m}",
        )
