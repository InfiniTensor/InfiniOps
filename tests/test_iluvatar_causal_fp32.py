import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("q,k", ((2, 3), (521, 521), (2051, 2051), (521, 4096)))
@pytest.mark.parametrize("inplace", (False, True))
def test_iluvatar_causal_fp32(q, k, inplace, device):
    if (
        device != "cuda"
        or not infini.ops.CausalSoftmaxInfinilm.active_implementation_indices(
            "iluvatar"
        )
    ):
        pytest.skip("requires the Iluvatar backend")

    # Include masked NaNs: the causal branch must select them away.
    index = torch.arange(2 * q * k, device=device).reshape(2, q, k)
    source = ((index % 29 - 14).float() / 8).to(torch.bfloat16)
    mask = torch.arange(k, device=device)[None, :] >= (
        k - q + torch.arange(q, device=device)[:, None] + 1
    )
    source.masked_fill_(mask, float("nan"))
    reference = source.clone().masked_fill_(mask, float("-inf"))
    # Match the FP32 sum before BF16 exponent storage of the InfiniLM path.
    infini.ops.softmax(reference, -1, None, reference, stream=get_stream(source.device))
    out = source if inplace else torch.empty_like(source)
    infini.ops.causal_softmax_infinilm(source, out, stream=get_stream(source.device))
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    assert torch.isfinite(out).all()
    assert torch.count_nonzero(out.masked_select(mask)) == 0
