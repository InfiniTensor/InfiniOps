import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("seq_len", (1, 17))
def test_fused_qk_norm_out_infinilm(seq_len, device, implementation_index):
    if device != "cuda" or implementation_index != 9:
        pytest.skip("NineToothed NVIDIA packed QKV output fast path")

    torch.manual_seed(7)
    dtype = torch.bfloat16
    query_heads, key_heads, value_heads = 16, 8, 8
    head_dim = 128
    qkv = torch.randn(
        (seq_len, (query_heads + key_heads + value_heads) * head_dim),
        device=device,
        dtype=dtype,
    )
    original_qkv = qkv.clone()
    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(head_dim, device=device, dtype=dtype)
    q_output = torch.empty(
        (seq_len, query_heads * head_dim), device=device, dtype=dtype
    )
    k_output = torch.empty((seq_len, key_heads * head_dim), device=device, dtype=dtype)

    infini.ops.fused_qk_norm_out_infinilm(
        qkv,
        query_heads,
        key_heads,
        value_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        q_output,
        k_output,
        implementation_index=implementation_index,
        stream=get_stream(device),
    )

    query = (
        qkv[:, : query_heads * head_dim].float().view(seq_len, query_heads, head_dim)
    )
    key = (
        qkv[:, query_heads * head_dim : (query_heads + key_heads) * head_dim]
        .float()
        .view(seq_len, key_heads, head_dim)
    )
    query = (
        query
        * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) / head_dim + 1e-6)
        * q_weight.float()
    )
    key = (
        key
        * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) / head_dim + 1e-6)
        * k_weight.float()
    )

    assert torch.equal(qkv, original_qkv)
    torch.testing.assert_close(
        q_output.float(),
        query.reshape(seq_len, -1).to(dtype).float(),
        rtol=2e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        k_output.float(),
        key.reshape(seq_len, -1).to(dtype).float(),
        rtol=2e-2,
        atol=1e-2,
    )
