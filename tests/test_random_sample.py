import infini.ops
import pytest
import torch

from tests.utils import Payload


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("vocab_size", (32, 2048, 73448))
@pytest.mark.parametrize(
    ("topk", "topp", "temperature", "random_val"),
    (
        (1, 1.0, 1.0, 0.5),
        (10, 1.0, 1.0, 0.123),
        (50, 0.9, 1.0, 0.7),
        (50, 0.9, 0.8, 0.0),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_random_sample(vocab_size, topk, topp, temperature, random_val, dtype, device):
    active_indices = infini.ops.RandomSample.active_implementation_indices(device)

    if 1 not in active_indices:
        pytest.skip(f"implementation `1` not active on `{device}`")

    logits = torch.randn(vocab_size, dtype=dtype, device=device)
    out = torch.empty((), dtype=torch.int64, device=device)

    return Payload(
        _random_sample,
        _torch_random_sample,
        (logits, random_val, topp, topk, temperature, out),
        {},
    )


def _random_sample(logits, random_val, topp, topk, temperature, out):
    infini.ops.random_sample(
        logits, random_val, topp, topk, temperature, out, implementation_index=1
    )

    return out


def _torch_random_sample(logits, random_val, topp, topk, temperature, out):
    if topk <= 1 or temperature == 0.0:
        out.copy_(torch.argmax(logits, dim=0))

        return out

    vocab_size = logits.size(0)
    effective_topk = min(topk, vocab_size)

    scaled = logits.to(torch.float32) / temperature
    probs = torch.softmax(scaled, dim=0)

    topk_probs, topk_indices = torch.topk(probs, effective_topk, dim=0)

    cumsum = torch.cumsum(topk_probs, dim=0)
    keep = cumsum < topp
    keep_prefix = torch.cat([torch.ones(1, dtype=keep.dtype, device=keep.device), keep[:-1]])
    filtered = torch.where(keep_prefix, topk_probs, torch.zeros_like(topk_probs))
    total = filtered.sum()

    if total.item() == 0.0:
        out.copy_(topk_indices[0])

        return out

    filtered = filtered / total
    cdf = torch.cumsum(filtered, dim=0)
    rv = torch.tensor(random_val, dtype=cdf.dtype, device=cdf.device)
    selected = torch.searchsorted(cdf, rv).clamp(0, effective_topk - 1)
    out.copy_(topk_indices[selected].to(out.dtype))

    return out
