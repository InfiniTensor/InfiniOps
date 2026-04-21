import math

import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, randn_strided


def _build_sin_cos_cache(max_seq_len, head_dim, theta, dtype, device):
    cache_dim = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)

    return angles.sin().to(dtype=dtype, device=device), angles.cos().to(dtype=dtype, device=device)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "batch, seq_len, num_heads, head_dim, max_seq_len",
    (
        (1, 8, 4, 32, 2048),
        (2, 16, 8, 64, 4096),
        (1, 128, 2, 64, 8192),
    ),
)
@pytest.mark.parametrize("is_neox_style", (True, False))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 5e-3, 5e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_rope(
    batch,
    seq_len,
    num_heads,
    head_dim,
    max_seq_len,
    is_neox_style,
    dtype,
    device,
    rtol,
    atol,
):
    active_indices = infini.ops.Rope.active_implementation_indices(device)

    if 1 not in active_indices:
        pytest.skip(f"implementation `1` not active on `{device}`")

    x = randn_strided((batch, seq_len, num_heads, head_dim), None, dtype=dtype, device=device)
    positions = torch.arange(seq_len, dtype=torch.int32, device=device).expand(batch, seq_len).contiguous()
    sin_cache, cos_cache = _build_sin_cos_cache(max_seq_len, head_dim, theta=10000.0, dtype=dtype, device=device)
    out = empty_strided(x.shape, None, dtype=dtype, device=device)

    return Payload(
        _rope,
        _torch_rope,
        (x, positions, sin_cache, cos_cache, is_neox_style, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _rope(x, positions, sin_cache, cos_cache, is_neox_style, out):
    infini.ops.rope(
        x, positions, sin_cache, cos_cache, is_neox_style, out, implementation_index=1
    )

    return out


def _torch_rope(x, positions, sin_cache, cos_cache, is_neox_style, out):
    head_dim = x.shape[-1]
    half = head_dim // 2

    pos_long = positions.to(torch.long)
    sin_at_pos = sin_cache[pos_long]
    cos_at_pos = cos_cache[pos_long]

    mid_dims = x.dim() - 1 - positions.dim()
    for _ in range(mid_dims):
        sin_at_pos = sin_at_pos.unsqueeze(-2)
        cos_at_pos = cos_at_pos.unsqueeze(-2)

    if is_neox_style:
        x1 = x[..., :half]
        x2 = x[..., half:]
        rot1 = x1 * cos_at_pos - x2 * sin_at_pos
        rot2 = x2 * cos_at_pos + x1 * sin_at_pos
        out[..., :half] = rot1
        out[..., half:] = rot2
    else:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos_at_pos - x_odd * sin_at_pos
        rot_odd = x_odd * cos_at_pos + x_even * sin_at_pos
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd

    return out


# Silences the "imported but unused" warning on `math`.
_ = math
