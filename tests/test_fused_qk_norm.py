import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("seq_len", (1, 17))
def test_fused_qk_norm(seq_len, device, implementation_index):
    if device != "cuda" or implementation_index != 9:
        pytest.skip("NineToothed NVIDIA packed QKV fast path")

    torch.manual_seed(7)
    dtype = torch.bfloat16
    query_heads, key_heads, value_heads = 16, 8, 8
    head_dim = 128
    qkv = torch.randn(
        (seq_len, (query_heads + key_heads + value_heads) * head_dim),
        device=device,
        dtype=dtype,
    )
    expected = qkv.clone()
    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(head_dim, device=device, dtype=dtype)

    infini.ops.fused_qk_norm(
        qkv,
        query_heads,
        key_heads,
        value_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        implementation_index=implementation_index,
        stream=get_stream(device),
    )

    query = (
        expected[:, : query_heads * head_dim]
        .view(seq_len, query_heads, head_dim)
        .float()
    )
    key = (
        expected[:, query_heads * head_dim : (query_heads + key_heads) * head_dim]
        .view(seq_len, key_heads, head_dim)
        .float()
    )
    query *= torch.rsqrt(query.square().sum(dim=-1, keepdim=True) / head_dim + 1e-6)
    key *= torch.rsqrt(key.square().sum(dim=-1, keepdim=True) / head_dim + 1e-6)
    query *= q_weight.float()
    key *= k_weight.float()
    expected = torch.cat(
        (
            query.reshape(seq_len, -1),
            key.reshape(seq_len, -1),
            expected[:, (query_heads + key_heads) * head_dim :],
        ),
        dim=-1,
    ).to(dtype)

    assert torch.equal(
        qkv[:, (query_heads + key_heads) * head_dim :],
        expected[:, (query_heads + key_heads) * head_dim :],
    )
    torch.testing.assert_close(qkv.float(), expected.float(), rtol=2e-2, atol=1e-2)
