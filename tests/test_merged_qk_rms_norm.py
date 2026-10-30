import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("seq_len", (1, 17))
def test_merged_qk_rms_norm(seq_len, device, implementation_index):
    if device != "cuda" or implementation_index != 9:
        pytest.skip("NineToothed NVIDIA packed Q/K fast path")

    torch.manual_seed(7)
    dtype = torch.bfloat16
    query_heads = 16
    key_heads = 8
    head_dim = 128
    projection = torch.randn((seq_len, 32, head_dim), device=device, dtype=dtype)
    input = projection[:, : query_heads + key_heads]
    query_weight = torch.randn(head_dim, device=device, dtype=dtype)
    key_weight = torch.randn(head_dim, device=device, dtype=dtype)
    weight = torch.cat(
        (query_weight.expand(query_heads, -1), key_weight.expand(key_heads, -1))
    )
    output = torch.empty(
        (seq_len, query_heads + key_heads, head_dim),
        device=device,
        dtype=dtype,
    )

    infini.ops.merged_qk_rms_norm(
        input,
        weight,
        1e-6,
        output,
        implementation_index=implementation_index,
        stream=get_stream(device),
    )

    expected = input.float()
    expected[:seq_len, :query_heads] *= query_weight.float()
    expected[:seq_len, query_heads:] *= key_weight.float()
    expected *= torch.rsqrt(
        input.float().square().sum(dim=-1, keepdim=True) / head_dim + 1e-6
    )
    torch.testing.assert_close(
        output.float(),
        expected.to(dtype).float(),
        rtol=2e-2,
        atol=1e-2,
    )
