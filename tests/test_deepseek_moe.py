import infini.ops
import pytest
import torch

from tests.utils import get_stream

if not hasattr(infini.ops, "DeepseekMoe"):
    pytest.skip(
        "`DeepseekMoe` is not available on this platform", allow_module_level=True
    )

# Small config so the reference stays trivially checkable while still
# exercising multiple experts, uneven per-expert loads and float32 weights.
# The 5000-token case crosses the 4096-token chunk boundary of the metax
# provider, so both single- and multi-chunk execution are covered.
_CASES = (
    ((5, 32, 3, 48, 6), torch.float16, 2e-3, 2e-3),
    ((5, 32, 3, 48, 6), torch.bfloat16, 4e-3, 4e-3),
    ((5000, 32, 3, 48, 6), torch.float16, 2e-3, 2e-3),
)


def _reference(hidden, topk_ids, topk_weights, gate, up, down):
    """MoE output = sum_k w[t,k] * ((silu(x @ g_e^T) * (x @ u_e^T)) @ d_e^T).

    Matches the provider contract: `topk_ids` selects the expert per route,
    `topk_weights` scales each route's contribution before the top-k sum.
    """
    out = torch.zeros_like(hidden)
    for token in range(hidden.size(0)):
        for k in range(topk_ids.size(1)):
            expert = int(topk_ids[token, k])
            weight = float(topk_weights[token, k])
            x = hidden[token : token + 1].float()
            gate_out = x @ gate[expert].t().float()
            up_out = x @ up[expert].t().float()
            activated = torch.nn.functional.silu(gate_out) * up_out * weight
            out[token] += (activated @ down[expert].t().float())[0]
    return out.to(hidden.dtype)


@pytest.mark.parametrize("shape_config, dtype, rtol, atol", _CASES)
def test_deepseek_moe(shape_config, dtype, device, rtol, atol):
    if device != "cuda":
        pytest.skip("deepseek_moe requires an accelerator device")

    ntokens, hidden_dim, topk, intermediate, num_experts = shape_config
    torch.manual_seed(42)
    # Native device construction avoids the MACA H2D stride bug; the expert
    # projections are scaled down so the output stays O(0.1) and rounding
    # noise stays far below the tolerance.
    hidden = torch.randn((ntokens, hidden_dim), dtype=dtype, device=device)
    logits = torch.randn((ntokens, num_experts), dtype=torch.float32, device=device)
    topk = torch.topk(logits, topk, dim=1)
    topk_ids = topk.indices.to(torch.int32)
    topk_weights = torch.softmax(topk.values, dim=1)
    gate = [
        torch.randn((intermediate, hidden_dim), dtype=dtype, device=device) * 0.1
        for _ in range(num_experts)
    ]
    up = [
        torch.randn((intermediate, hidden_dim), dtype=dtype, device=device) * 0.1
        for _ in range(num_experts)
    ]
    down = [
        torch.randn((hidden_dim, intermediate), dtype=dtype, device=device) * 0.1
        for _ in range(num_experts)
    ]
    out = torch.full((ntokens, hidden_dim), torch.nan, dtype=dtype, device=device)

    result = infini.ops.deepseek_moe(
        out,
        hidden,
        topk_ids,
        topk_weights,
        gate,
        up,
        down,
        intermediate,
        num_experts,
        stream=get_stream(hidden.device),
    )

    assert result is None
    expected = _reference(hidden, topk_ids, topk_weights, gate, up, down)
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
