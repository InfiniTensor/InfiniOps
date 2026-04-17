import infini.ops
import pytest
import torch

from tests.utils import get_stream


def _topk_unsupported_reason():
    """Return a reason string if `TopkToppSampling` cannot run here.

    The ATB `TopkToppSamplingParam` kernel fails with a vector core exception
    (ACL error `507035`, `[TopkToppSampling] Execute failed (status=1)`) when
    executed with varying `(batch, vocab)` shapes in sequence on Ascend 910B.
    Only the first shape succeeds; every subsequent call crashes the stream.
    Reproduced on `feat/ascend-operators` (full op set, pre-existing) with
    identical kernel — not a split-branch regression.  Pending an upstream
    ATB executor fix, skip on 910B.
    """
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return ""

    name = torch.npu.get_device_name(0)

    if "910B" in name:
        return (
            f"pre-existing ATB `TopkToppSamplingParam` kernel bug on {name}: "
            "vector core exception (ACL 507035) when executed with different "
            "(batch, vocab) shapes in sequence"
        )

    return ""


_skip_topk = pytest.mark.skipif(
    bool(_topk_unsupported_reason()),
    reason=_topk_unsupported_reason() or "TopkToppSampling unsupported",
)


@_skip_topk
@pytest.mark.parametrize(
    "batch_size, vocab_size",
    (
        (1, 128),
        (4, 1024),
        (8, 32000),
    ),
)
@pytest.mark.parametrize("topk", (1, 8, 64))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("device", ("npu",))
def test_topk_topp_sampling(batch_size, vocab_size, topk, dtype, device):
    """Verify ATB top-k sampling returns an index in the top-k set.

    The kernel uses exponential sampling (Gumbel-trick) which is
    non-deterministic, so we cannot compare against a fixed reference.
    Instead we assert the sampled token is among the `topk` highest-
    probability tokens for each row.
    """
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    active_indices = infini.ops.TopkToppSampling.active_implementation_indices(device)

    if not active_indices:
        pytest.skip(f"TopkToppSampling not registered for `{device}`")

    if topk > vocab_size:
        pytest.skip(f"`topk={topk}` exceeds `vocab_size={vocab_size}`")

    logits = torch.randn((batch_size, vocab_size), dtype=torch.float32, device=device)
    probs = torch.softmax(logits, dim=-1).to(dtype)
    out = torch.empty((batch_size,), dtype=torch.int32, device=device)

    infini.ops.topk_topp_sampling(
        probs,
        topk,
        0.0,
        out,
        stream=get_stream(probs.device),
    )

    _, topk_indices = torch.topk(probs.to(torch.float32), k=topk, dim=-1)
    topk_set = topk_indices.cpu()
    sampled = out.to(torch.int64).cpu()

    for i in range(batch_size):
        assert sampled[i].item() in set(topk_set[i].tolist()), (
            f"row {i}: sampled token `{sampled[i].item()}` not in "
            f"top-{topk} set `{topk_set[i].tolist()}`"
        )
