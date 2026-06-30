import infini.ops
import pytest
import torch

from tests.utils import get_stream, randn_strided, randint_strided


@pytest.fixture(autouse=True)
def _clear_rotary_cache():
    infini.ops.RotaryEmbedding.clear_cache()

    yield


@pytest.mark.parametrize(
    "num_heads, head_size",
    (
        (32, 128),
        (8, 64),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-2),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
@pytest.mark.parametrize("device", ("npu",))
def test_rotary_embedding(num_heads, head_size, dtype, rtol, atol, device):
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    num_tokens = 16
    num_kv_heads = num_heads
    rotary_dim = head_size
    max_seq_len = 64

    positions = randint_strided(
        0,
        max_seq_len,
        (num_tokens,),
        None,
        dtype=torch.int64,
        device=device,
    )
    query = randn_strided(
        (num_tokens, num_heads, head_size),
        None,
        dtype=dtype,
        device=device,
    )
    key = randn_strided(
        (num_tokens, num_kv_heads, head_size),
        None,
        dtype=dtype,
        device=device,
    )
    cos_sin_cache = randn_strided(
        (max_seq_len, rotary_dim),
        None,
        dtype=dtype,
        device=device,
    )
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)

    infini.ops.rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        rotary_dim,
        True,
        query_out,
        key_out,
        stream=get_stream(query.device),
    )

    ref_q, ref_k = _ref_rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        rotary_dim,
    )

    _assert_close(query_out, ref_q, rtol, atol)
    _assert_close(key_out, ref_k, rtol, atol)


def _ref_rotary_embedding(positions, query, key, cos_sin_cache, head_size, rotary_dim):
    num_tokens = query.size(0)
    half_rotary_dim = rotary_dim // 2
    out_q = query.float().clone()
    out_k = key.float().clone()

    cos_sin = cos_sin_cache.float()
    cos_half = cos_sin[:, :half_rotary_dim]
    sin_half = cos_sin[:, half_rotary_dim:]

    def apply_rope(input_tensor, output_tensor):
        for token_idx in range(num_tokens):
            position = positions[token_idx].item()
            cos = cos_half[position]
            sin = sin_half[position]

            x1 = input_tensor[token_idx, :, :half_rotary_dim].float()
            x2 = input_tensor[token_idx, :, half_rotary_dim:rotary_dim].float()
            output_tensor[token_idx, :, :half_rotary_dim] = cos * x1 - sin * x2
            output_tensor[token_idx, :, half_rotary_dim:rotary_dim] = (
                cos * x2 + sin * x1
            )

    apply_rope(query, out_q)
    apply_rope(key, out_k)

    return out_q.to(query.dtype), out_k.to(key.dtype)


def _assert_close(actual, expected, rtol, atol):
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Max diff: {(actual.float() - expected.float()).abs().max().item()}"
    )
