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


@pytest.mark.parametrize("padded_batch", (False, True))
def test_fused_qk_norm_native(padded_batch, device, implementation_index):
    """Native CUDA packed provider. When the packed rows are carved from
    a padded batch the real token stride exceeds the row width, and the
    kernel must walk tokens with that stride instead of the row width."""
    if device != "cuda" or implementation_index != 0:
        pytest.skip("native CUDA packed QKV norm provider")

    torch.manual_seed(6)
    dtype = torch.bfloat16
    query_heads, key_heads, value_heads, head_dim = 16, 8, 8, 128
    row_width = (query_heads + key_heads + value_heads) * head_dim
    if padded_batch:
        source = torch.randn((2, 20, row_width + 11), device=device, dtype=dtype)
        qkv = source[:, :, :row_width].view(40, row_width)
        seq_len = 40
    else:
        qkv = torch.randn((17, row_width), device=device, dtype=dtype)
        seq_len = 17
    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(head_dim, device=device, dtype=dtype)
    # The op normalizes `qkv` in place; snapshot the input for the reference.
    original = qkv.clone()

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
        original[:, : query_heads * head_dim]
        .reshape(seq_len, query_heads, head_dim)
        .float()
    )
    key = (
        original[:, query_heads * head_dim : (query_heads + key_heads) * head_dim]
        .reshape(seq_len, key_heads, head_dim)
        .float()
    )
    expected_query = query * torch.rsqrt(
        query.square().sum(dim=-1, keepdim=True) / head_dim + 1e-6
    )
    expected_key = key * torch.rsqrt(
        key.square().sum(dim=-1, keepdim=True) / head_dim + 1e-6
    )
    expected_query *= q_weight.float()
    expected_key *= k_weight.float()
    assert torch.isfinite(qkv).all()
    torch.testing.assert_close(
        qkv[:, : query_heads * head_dim],
        expected_query.reshape(seq_len, -1).to(dtype),
        rtol=2e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        qkv[:, query_heads * head_dim : (query_heads + key_heads) * head_dim],
        expected_key.reshape(seq_len, -1).to(dtype),
        rtol=2e-2,
        atol=1e-2,
    )
