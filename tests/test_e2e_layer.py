import infini.ops
import pytest
import torch

from tests.utils import get_npu_stream, randn_strided, randint_strided


def _stream_kw(tensor):
    if tensor.device.type == "npu":
        return {"stream": get_npu_stream(tensor)}

    return {}


def _ref_rms_norm(x, weight, eps):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)

    return (x / rms) * weight


def _ref_rope(positions, query, key, cos_sin_cache, head_size, rotary_dim,
              is_neox_style):
    T = query.size(0)
    R = rotary_dim
    half_R = R // 2
    cos_half = cos_sin_cache[:, :half_R]
    sin_half = cos_sin_cache[:, half_R:]

    def apply_rope(x):
        out = x.clone()

        for t in range(T):
            p = positions[t].item()
            c = cos_half[p]
            s = sin_half[p]

            if is_neox_style:
                x1 = x[t, :, :half_R]
                x2 = x[t, :, half_R:R]
                out[t, :, :half_R] = c * x1 - s * x2
                out[t, :, half_R:R] = c * x2 + s * x1
            else:
                x1 = x[t, :, 0::2]
                x2 = x[t, :, 1::2]
                out[t, :, 0::2] = c * x1 - s * x2
                out[t, :, 1::2] = c * x2 + s * x1

        return out

    return apply_rope(query), apply_rope(key)


def _ref_sdpa(query, key, value, num_heads, num_kv_heads, head_size, scale,
              causal):
    q = query.transpose(0, 1).float()
    k = key.transpose(0, 1).float()
    v = value.transpose(0, 1).float()

    if num_kv_heads < num_heads:
        ratio = num_heads // num_kv_heads
        k = k.repeat_interleave(ratio, dim=0)
        v = v.repeat_interleave(ratio, dim=0)

    out = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
        scale=scale, is_causal=causal,
    )

    return out.squeeze(0).transpose(0, 1)


def _infiniops_layer(
    hidden, positions, cos_sin_cache,
    input_norm_w, qkv_proj_w, o_proj_w,
    gate_proj_w, up_proj_w, down_proj_w,
    post_norm_w,
    num_heads, num_kv_heads, head_size,
    rotary_dim, intermediate_size,
    is_neox_style, eps, scale, num_tokens,
):
    """Run one LLaMA decoder layer using InfiniOps kernels."""
    kw = _stream_kw(hidden)
    dtype = hidden.dtype
    device = hidden.device
    hidden_size = hidden.size(-1)

    # Save residual.
    residual = hidden.clone()

    # 1. Input RMSNorm.
    normed = torch.empty_like(hidden)
    infini.ops.rms_norm(hidden, input_norm_w, eps, normed, **kw)

    # 2. QKV projection: [T, D] @ [D, (N+2*Nkv)*H] -> [T, (N+2*Nkv)*H].
    qkv_dim = (num_heads + 2 * num_kv_heads) * head_size
    qkv = torch.empty(num_tokens, qkv_dim, dtype=dtype, device=device)
    infini.ops.gemm(normed, qkv_proj_w, 1.0, 0.0, False, False, qkv, **kw)

    # Split Q, K, V.
    q = qkv[:, :num_heads * head_size].reshape(
        num_tokens, num_heads, head_size,
    ).contiguous()
    k = qkv[:, num_heads * head_size:(num_heads + num_kv_heads) * head_size].reshape(
        num_tokens, num_kv_heads, head_size,
    ).contiguous()
    v = qkv[:, (num_heads + num_kv_heads) * head_size:].reshape(
        num_tokens, num_kv_heads, head_size,
    ).contiguous()

    # 3. RoPE.
    q_rot = torch.empty_like(q)
    k_rot = torch.empty_like(k)
    infini.ops.rotary_embedding(
        positions, q, k, cos_sin_cache,
        head_size, rotary_dim, is_neox_style,
        q_rot, k_rot, **kw,
    )

    # 4. Flash attention (single-sequence prefill, causal).
    attn_out = torch.empty(
        num_tokens, num_heads, head_size, dtype=dtype, device=device,
    )
    infini.ops.flash_attention(
        q_rot, k_rot, v,
        None, None, None,
        num_heads, num_kv_heads, head_size,
        scale, True, -1, 0, 0, attn_out, **kw,
    )

    # 5. O projection: [T, N*H] @ [N*H, D] -> [T, D].
    attn_2d = attn_out.reshape(num_tokens, num_heads * head_size)
    o_out = torch.empty(num_tokens, hidden_size, dtype=dtype, device=device)
    infini.ops.gemm(attn_2d, o_proj_w, 1.0, 0.0, False, False, o_out, **kw)

    # 6. Residual add.
    after_attn = torch.empty_like(residual)
    infini.ops.add(residual, o_out, after_attn, **kw)

    # 7. Post-attention RMSNorm.
    residual2 = after_attn.clone()
    normed2 = torch.empty_like(after_attn)
    infini.ops.rms_norm(after_attn, post_norm_w, eps, normed2, **kw)

    # 8. Gate + up projections.
    gate = torch.empty(num_tokens, intermediate_size, dtype=dtype, device=device)
    up = torch.empty(num_tokens, intermediate_size, dtype=dtype, device=device)
    infini.ops.gemm(normed2, gate_proj_w, 1.0, 0.0, False, False, gate, **kw)
    infini.ops.gemm(normed2, up_proj_w, 1.0, 0.0, False, False, up, **kw)

    # 9. SwiGLU: ``up * silu(gate)``.
    ffn = torch.empty(num_tokens, intermediate_size, dtype=dtype, device=device)
    infini.ops.swiglu(up, gate, ffn, **kw)

    # 10. Down projection: [T, FFN] @ [FFN, D] -> [T, D].
    down = torch.empty(num_tokens, hidden_size, dtype=dtype, device=device)
    infini.ops.gemm(ffn, down_proj_w, 1.0, 0.0, False, False, down, **kw)

    # 11. Second residual add.
    output = torch.empty_like(residual2)
    infini.ops.add(residual2, down, output, **kw)

    return output


def _reference_layer(
    hidden, positions, cos_sin_cache,
    input_norm_w, qkv_proj_w, o_proj_w,
    gate_proj_w, up_proj_w, down_proj_w,
    post_norm_w,
    num_heads, num_kv_heads, head_size,
    rotary_dim, intermediate_size,
    is_neox_style, eps, scale, num_tokens,
):
    """PyTorch float32 reference for one LLaMA decoder layer."""
    hidden_size = hidden.size(-1)

    # Compute in float32 on CPU for accuracy.
    h = hidden.float().cpu()
    pos = positions.cpu()
    csc = cos_sin_cache.float().cpu()
    inw = input_norm_w.float().cpu()
    qkvw = qkv_proj_w.float().cpu()
    ow = o_proj_w.float().cpu()
    gw = gate_proj_w.float().cpu()
    uw = up_proj_w.float().cpu()
    dw = down_proj_w.float().cpu()
    pnw = post_norm_w.float().cpu()

    # 1. Input RMSNorm.
    residual = h.clone()
    normed = _ref_rms_norm(h, inw, eps)

    # 2. QKV projection.
    qkv_dim = (num_heads + 2 * num_kv_heads) * head_size
    qkv = normed @ qkvw

    q = qkv[:, :num_heads * head_size].reshape(num_tokens, num_heads, head_size)
    k = qkv[:, num_heads * head_size:(num_heads + num_kv_heads) * head_size].reshape(
        num_tokens, num_kv_heads, head_size,
    )
    v = qkv[:, (num_heads + num_kv_heads) * head_size:].reshape(
        num_tokens, num_kv_heads, head_size,
    )

    # 3. RoPE.
    q_rot, k_rot = _ref_rope(
        pos, q, k, csc, head_size, rotary_dim, is_neox_style,
    )

    # 4. SDPA.
    attn_out = _ref_sdpa(q_rot, k_rot, v, num_heads, num_kv_heads, head_size,
                         scale, causal=True)

    # 5. O projection.
    attn_2d = attn_out.reshape(num_tokens, num_heads * head_size)
    o_out = attn_2d @ ow

    # 6. Residual add.
    after_attn = residual + o_out

    # 7. Post-attention RMSNorm.
    residual2 = after_attn.clone()
    normed2 = _ref_rms_norm(after_attn, pnw, eps)

    # 8. Gate + up projections.
    gate = normed2 @ gw
    up = normed2 @ uw

    # 9. SwiGLU: ``up * silu(gate)``.
    ffn = up * (gate * torch.sigmoid(gate))

    # 10. Down projection.
    down = ffn @ dw

    # 11. Second residual add.
    output = residual2 + down

    return output.to(hidden.dtype).to(hidden.device)


def _make_rope_cache(max_seq_len, rotary_dim, dtype, device):
    """Build a proper RoPE cos/sin cache (bounded to [-1, 1])."""
    half_dim = rotary_dim // 2
    freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freq)  # [max_seq_len, half_dim]
    cos_half = torch.cos(angles).to(dtype=dtype, device=device)
    sin_half = torch.sin(angles).to(dtype=dtype, device=device)

    return torch.cat([cos_half, sin_half], dim=-1)


@pytest.mark.parametrize("device", ("npu",))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 5e-3, 5e-3),
        (torch.bfloat16, 1e-2, 2e-2),
    ),
)
def test_llama_layer(device, dtype, rtol, atol):
    """End-to-end test of a LLaMA decoder layer using InfiniOps kernels."""
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    # Small LLaMA-like model config.
    hidden_size = 512
    num_heads = 8
    num_kv_heads = 2
    head_size = hidden_size // num_heads
    intermediate_size = 1024
    num_tokens = 1
    max_seq_len = 16
    rotary_dim = head_size
    is_neox_style = True
    eps = 1e-6
    scale = 1.0 / head_size**0.5

    def _scaled_weight(*shape):
        return randn_strided(shape, None, dtype=dtype, device=device) / shape[0] ** 0.5

    # Random weights (stored as [in_features, out_features], Xavier-scaled).
    qkv_proj_w = _scaled_weight(
        hidden_size, (num_heads + 2 * num_kv_heads) * head_size,
    )
    o_proj_w = _scaled_weight(num_heads * head_size, hidden_size)
    gate_proj_w = _scaled_weight(hidden_size, intermediate_size)
    up_proj_w = _scaled_weight(hidden_size, intermediate_size)
    down_proj_w = _scaled_weight(intermediate_size, hidden_size)
    input_norm_w = torch.ones(hidden_size, dtype=dtype, device=device)
    post_norm_w = torch.ones(hidden_size, dtype=dtype, device=device)

    # Proper cos/sin cache from frequency decomposition (bounded [-1, 1]).
    cos_sin_cache = _make_rope_cache(max_seq_len, rotary_dim, dtype, device)
    positions = randint_strided(
        0, max_seq_len, (num_tokens,), None,
        dtype=torch.int64, device=device,
    )

    # Input hidden states scaled to prevent value explosion through layers.
    hidden = randn_strided(
        (num_tokens, hidden_size), None, dtype=dtype, device=device,
    ) / hidden_size**0.5

    common = dict(
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        input_norm_w=input_norm_w,
        qkv_proj_w=qkv_proj_w,
        o_proj_w=o_proj_w,
        gate_proj_w=gate_proj_w,
        up_proj_w=up_proj_w,
        down_proj_w=down_proj_w,
        post_norm_w=post_norm_w,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rotary_dim=rotary_dim,
        intermediate_size=intermediate_size,
        is_neox_style=is_neox_style,
        eps=eps,
        scale=scale,
        num_tokens=num_tokens,
    )

    infini_out = _infiniops_layer(hidden, **common)
    ref_out = _reference_layer(hidden, **common)

    max_diff = (infini_out.float() - ref_out.float()).abs().max().item()
    assert torch.allclose(infini_out, ref_out, rtol=rtol, atol=atol), (
        f"Max diff: {max_diff}"
    )
