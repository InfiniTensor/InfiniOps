import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("seq_len", (1, 17))
def test_fused_qk_rms_norm(seq_len, device, implementation_index):
    if device != "cuda" or implementation_index != 9:
        pytest.skip("NineToothed NVIDIA separate Q/K RMSNorm fast path")

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
    q = qkv[:, : query_heads * head_dim].view(seq_len, query_heads, head_dim)
    kv = qkv[:, query_heads * head_dim : (query_heads + key_heads) * head_dim].view(
        seq_len, key_heads, head_dim
    )
    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    kv_weight = torch.randn(head_dim, device=device, dtype=dtype)
    q_output = torch.empty((seq_len, query_heads, head_dim), device=device, dtype=dtype)
    kv_output = torch.empty((seq_len, key_heads, head_dim), device=device, dtype=dtype)

    infini.ops.fused_qk_rms_norm(
        q,
        kv,
        q_weight,
        kv_weight,
        1e-6,
        q_output,
        kv_output,
        implementation_index=implementation_index,
        stream=get_stream(device),
    )

    expected_q = (
        q.float()
        * torch.rsqrt(q.float().square().sum(dim=-1, keepdim=True) / head_dim + 1e-6)
        * q_weight.float()
    )
    expected_kv = (
        kv.float()
        * torch.rsqrt(kv.float().square().sum(dim=-1, keepdim=True) / head_dim + 1e-6)
        * kv_weight.float()
    )

    assert torch.equal(qkv, original_qkv)
    torch.testing.assert_close(
        q_output.float(),
        expected_q.to(dtype).float(),
        rtol=2e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        kv_output.float(),
        expected_kv.to(dtype).float(),
        rtol=2e-2,
        atol=1e-2,
    )


@pytest.mark.parametrize("padded_source", (False, True))
@pytest.mark.parametrize("padded_output", (False, True))
def test_fused_qk_rms_norm_native_strided(
    padded_source,
    padded_output,
    device,
    implementation_index,
):
    """Native CUDA provider over strided views carved out of a packed
    qkv projection: q/kv keep their real (possibly padded) token and
    head strides, which the kernel must honor row by row."""
    if device != "cuda" or implementation_index != 0:
        pytest.skip("native CUDA separate Q/K RMSNorm provider")

    torch.manual_seed(5)
    dtype = torch.bfloat16
    seq_len, query_heads, key_heads, head_dim = 17, 16, 8, 128
    qkv = torch.randn(
        (seq_len, (query_heads + key_heads) * head_dim),
        device=device,
        dtype=dtype,
    )
    q_width = query_heads * head_dim
    q = qkv[:, :q_width].view(seq_len, query_heads, head_dim)
    kv = qkv[:, q_width:].view(seq_len, key_heads, head_dim)
    if padded_source:
        q_source = torch.randn(
            (seq_len, query_heads, head_dim + 9), device=device, dtype=dtype
        )
        kv_source = torch.randn(
            (seq_len, key_heads, head_dim + 7), device=device, dtype=dtype
        )
        q = q_source[:, :, :head_dim]
        kv = kv_source[:, :, :head_dim]

    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    kv_weight = torch.randn(head_dim, device=device, dtype=dtype)
    if padded_output:
        q_output = torch.randn(
            (seq_len, query_heads, head_dim + 3), device=device, dtype=dtype
        )[:, :, :head_dim]
        kv_output = torch.randn(
            (seq_len, key_heads, head_dim + 5), device=device, dtype=dtype
        )[:, :, :head_dim]
    else:
        q_output = torch.empty(
            (seq_len, query_heads, head_dim), device=device, dtype=dtype
        )
        kv_output = torch.empty(
            (seq_len, key_heads, head_dim), device=device, dtype=dtype
        )

    infini.ops.fused_qk_rms_norm(
        q,
        kv,
        q_weight,
        kv_weight,
        1e-6,
        q_output,
        kv_output,
        implementation_index=implementation_index,
        stream=get_stream(device),
    )

    expected_q = (
        q.float()
        * torch.rsqrt(q.float().square().sum(dim=-1, keepdim=True) / head_dim + 1e-6)
        * q_weight.float()
    )
    expected_kv = (
        kv.float()
        * torch.rsqrt(kv.float().square().sum(dim=-1, keepdim=True) / head_dim + 1e-6)
        * kv_weight.float()
    )
    torch.testing.assert_close(
        q_output.float(),
        expected_q.to(dtype).float(),
        rtol=2e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        kv_output.float(),
        expected_kv.to(dtype).float(),
        rtol=2e-2,
        atol=1e-2,
    )
