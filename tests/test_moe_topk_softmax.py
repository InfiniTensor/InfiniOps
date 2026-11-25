import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infini.ops
import pytest
import torch
import torch.nn.functional as F

from tests.utils import empty_strided, get_stream


def _ref_moe_topk_softmax(gating_output, correction_bias, topk, renormalize, moe_softcapping):
    """Reference implementation using PyTorch."""
    # Convert to float32 for computation
    gating_fp32 = gating_output.float()

    # Apply softcapping
    if moe_softcapping != 0.0:
        gating_fp32 = torch.tanh(gating_fp32 / moe_softcapping) * moe_softcapping

    # Softmax (bias is applied *after* softmax in the kernel)
    probs = F.softmax(gating_fp32, dim=-1)

    # Top-k with optional post-softmax bias correction.
    # The SGLang kernel adds correction_bias to probs for selection,
    # but returns the original softmax probs as weights.
    if correction_bias is not None:
        selection_scores = probs + correction_bias.unsqueeze(0)
    else:
        selection_scores = probs
    topk_weights, topk_indices = torch.topk(selection_scores, topk, dim=-1)

    # Gather the original softmax probs at the selected indices
    topk_weights = torch.gather(probs, dim=-1, index=topk_indices)

    # Renormalize if requested
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_indices.to(torch.int32)


@pytest.mark.parametrize(
    "num_tokens, num_experts, topk, renormalize, moe_softcapping, has_bias",
    (
        (4, 8, 2, True, 0.0, False),
        (8, 16, 4, False, 50.0, False),
        (8, 16, 4, True, 50.0, True),
        (1, 4, 1, True, 0.0, False),
        (16, 64, 6, True, 30.0, True),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-1, 5e-2),
    ),
)
def test_moe_topk_softmax(
    num_tokens,
    num_experts,
    topk,
    renormalize,
    moe_softcapping,
    has_bias,
    dtype,
    device,
    rtol,
    atol,
):
    gating_output = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)

    topk_weights = empty_strided(
        (num_tokens, topk), None, dtype=torch.float32, device=device
    )
    topk_indices = empty_strided(
        (num_tokens, topk), None, dtype=torch.int32, device=device
    )

    if has_bias:
        correction_bias = torch.randn(num_experts, dtype=torch.float32, device=device)
    else:
        correction_bias = torch.empty(0, dtype=torch.float32, device=device)

    infini.ops.moe_topk_softmax(
        topk_weights,
        topk_indices,
        gating_output,
        correction_bias,
        renormalize,
        moe_softcapping,
        stream=get_stream(device),
    )

    ref_weights, ref_indices = _ref_moe_topk_softmax(
        gating_output, correction_bias if has_bias else None, topk, renormalize, moe_softcapping
    )

    torch.testing.assert_close(topk_weights, ref_weights, rtol=rtol, atol=atol)
    torch.testing.assert_close(topk_indices, ref_indices, rtol=0, atol=0)
