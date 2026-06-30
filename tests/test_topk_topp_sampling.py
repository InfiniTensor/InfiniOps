import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("shape", ((1, 8), (3, 16)))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_topk_topp_sampling(
    shape,
    dtype,
    device,
    implementation_index,
):
    batch_size, vocab_size = shape
    probs = torch.full(shape, 1e-3, dtype=dtype, device=device)

    for i in range(batch_size):
        probs[i, i % vocab_size] = 1.0

    probs = probs / probs.sum(dim=-1, keepdim=True)
    out = torch.empty((batch_size,), dtype=torch.int32, device=device)

    return Payload(
        _topk_topp_sampling,
        _torch_argmax,
        (probs, out),
        {"topk": 1, "topp": 1.0, "implementation_index": implementation_index},
    )


def _topk_topp_sampling(probs, out, *, topk, topp, implementation_index):
    infini.ops.topk_topp_sampling(
        probs,
        topk,
        topp,
        out,
        stream=get_stream(probs.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_argmax(probs, out, *, topk, topp, implementation_index):
    del topk, topp, implementation_index

    out.copy_(torch.argmax(probs, dim=-1).to(torch.int32))

    return out
