import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("seq_len", (1, 17))
@pytest.mark.parametrize("rotary_ratio", (1.0,))
@pytest.mark.parametrize("forced_token_heads_per_warp", (None, -1))
def test_fused_qk_norm_rope(
    seq_len, rotary_ratio, forced_token_heads_per_warp, device, implementation_index
):
    if device != "cuda" or implementation_index != 9:
        pytest.skip("NineToothed NVIDIA packed QKV fast path")

    torch.manual_seed(7)
    dtype = torch.bfloat16
    query_heads = 16
    key_heads = 8
    value_heads = 8
    head_dim = 128
    rotary_dim = int(head_dim * rotary_ratio)
    half_rotary = rotary_dim // 2
    total_dim = (query_heads + key_heads + value_heads) * head_dim

    qkv = torch.randn((seq_len, total_dim), device=device, dtype=dtype)
    expected = qkv.clone()
    q_weight = torch.randn(head_dim, device=device, dtype=dtype)
    k_weight = torch.randn(head_dim, device=device, dtype=dtype)
    cos_sin_cache = torch.randn((32, rotary_dim), device=device, dtype=dtype)
    positions = torch.randint(
        0, cos_sin_cache.size(0), (seq_len,), device=device, dtype=torch.int64
    )

    infini.ops.fused_qk_norm_rope(
        qkv,
        query_heads,
        key_heads,
        value_heads,
        head_dim,
        1e-6,
        q_weight,
        k_weight,
        cos_sin_cache,
        True,
        positions,
        *(
            ()
            if forced_token_heads_per_warp is None
            else (forced_token_heads_per_warp,)
        ),
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
    value = expected[:, (query_heads + key_heads) * head_dim :].view(
        seq_len, value_heads, head_dim
    )

    query *= torch.rsqrt(query.square().sum(dim=-1, keepdim=True) / head_dim + 1e-6)
    key *= torch.rsqrt(key.square().sum(dim=-1, keepdim=True) / head_dim + 1e-6)
    query *= q_weight.float()
    key *= k_weight.float()
    is_neox = True

    def apply_rope(tensor):
        if is_neox:
            first = tensor[..., :half_rotary]
            second = tensor[..., half_rotary:rotary_dim]
            first_index = torch.arange(half_rotary, device=tensor.device)
            second_index = first_index + half_rotary
        else:
            first = tensor[..., 0:rotary_dim:2]
            second = tensor[..., 1:rotary_dim:2]
            first_index = torch.arange(0, rotary_dim, 2, device=tensor.device)
            second_index = first_index + 1

        cos = (
            cos_sin_cache[positions[:, None], first_index[None, :]].float().unsqueeze(1)
        )
        sin = (
            cos_sin_cache[positions[:, None], second_index[None, :]]
            .float()
            .unsqueeze(1)
        )
        rotated_first = first * cos - second * sin
        rotated_second = second * cos + first * sin
        tensor = tensor.clone()
        if is_neox:
            tensor[..., :half_rotary] = rotated_first
            tensor[..., half_rotary:rotary_dim] = rotated_second
        else:
            tensor[..., 0:rotary_dim:2] = rotated_first
            tensor[..., 1:rotary_dim:2] = rotated_second
        return tensor

    query = apply_rope(query)
    key = apply_rope(key)
    expected = torch.cat(
        (
            query.reshape(seq_len, -1),
            key.reshape(seq_len, -1),
            value.reshape(seq_len, -1),
        ),
        dim=-1,
    ).to(dtype)

    torch.testing.assert_close(qkv.float(), expected.float(), rtol=2e-2, atol=1e-2)
