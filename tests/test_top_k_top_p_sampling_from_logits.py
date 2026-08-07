import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize(
    "top_k_value, top_p_value, allowed",
    (
        (3, 1.0, (0, 1, 2)),
        (16, 0.6, (0,)),
        (3, 0.8, (0, 1)),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_top_k_top_p_sampling_from_logits(
    top_k_value,
    top_p_value,
    allowed,
    dtype,
    device,
    implementation_index,
):
    batch_size = 64
    logits = torch.full((batch_size, 16), -10.0, dtype=dtype, device=device)
    logits[:, 0] = 5.0
    logits[:, 1] = 4.0
    logits[:, 2] = 3.0
    top_k = torch.full((batch_size,), top_k_value, dtype=torch.int64)
    top_p = torch.full((batch_size,), top_p_value, dtype=torch.float32)
    first = torch.empty((batch_size,), dtype=torch.int32, device=device)
    second = torch.empty_like(first)
    different_seed = torch.empty_like(first)

    _top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, 1234, 9, first, implementation_index
    )
    _top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, 1234, 9, second, implementation_index
    )
    _top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, 5678, 9, different_seed, implementation_index
    )

    assert torch.equal(first, second)
    if len(allowed) > 1:
        assert not torch.equal(first, different_seed)
    allowed_tensor = torch.tensor(allowed, dtype=torch.int32, device=device)
    assert torch.all(torch.isin(first, allowed_tensor))


def _top_k_top_p_sampling_from_logits(
    logits,
    top_k,
    top_p,
    seed,
    offset,
    out,
    implementation_index,
):
    infini.ops.top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        None,
        "top_k_first",
        True,
        False,
        seed,
        offset,
        out,
        stream=get_stream(logits.device),
        implementation_index=implementation_index,
    )
