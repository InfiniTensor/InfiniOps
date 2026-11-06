import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("batch", (1, 2))
@pytest.mark.parametrize("length", (7, 2049, 2910, 4096))
@pytest.mark.parametrize("implementation_index", (0,))
def test_iluvatar_decode_reduction(batch, length, device, implementation_index):
    if (
        device != "cuda"
        or not infini.ops.PagedAttentionInfinilm.active_implementation_indices(
            "iluvatar"
        )
    ):
        pytest.skip("requires the Iluvatar backend")

    generator = torch.Generator(device=device).manual_seed(20261105)
    q = torch.randn((batch, 32, 128), generator=generator, device=device).to(
        torch.bfloat16
    )
    k = torch.randn((batch, 2, 4096, 128), generator=generator, device=device).to(
        torch.bfloat16
    )
    v = torch.randn((batch, 2, 4096, 128), generator=generator, device=device).to(
        torch.bfloat16
    )
    blocks = torch.arange(batch, device=device, dtype=torch.int32).reshape(batch, 1)
    lengths = torch.tensor(
        [length - i for i in range(batch)], dtype=torch.int32, device=device
    )
    out = torch.empty_like(q)
    infini.ops.paged_attention_infinilm(
        q, k, v, blocks, lengths, None, 128**-0.5, out, stream=get_stream(q.device),
        implementation_index=implementation_index,
    )
    expected = torch.empty_like(q)
    for b in range(batch):
        n = length - b
        keys = k[b, :, :n].float().repeat_interleave(16, dim=0)
        values = v[b, :, :n].float().repeat_interleave(16, dim=0)
        scores = torch.einsum("hd,htd->ht", q[b].float(), keys) * (128**-0.5)
        result = torch.einsum("ht,htd->hd", torch.softmax(scores, -1), values)
        expected[b].copy_(result.to(torch.bfloat16))
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-3)
    assert torch.isfinite(out).all()
