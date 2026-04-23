import infini.ops
import pytest
import torch

from tests.utils import get_stream, randn_strided, randint_strided


@pytest.fixture(autouse=True)
def _clear_rotary_cache():
    """Clear the `RotaryEmbedding` op cache before each test.

    `CacheKey` ignores the `cos_sin_cache` data pointer, so a cached op
    constructed by a previous test with different cache contents would be
    reused here.  In production vLLM inference the cache is loaded once,
    so this pollution is a test-only hazard.
    """
    infini.ops.RotaryEmbedding.clear_cache()

    yield


def _rotary_embedding(
    positions,
    query,
    key,
    cos_sin_cache,
    head_size,
    rotary_dim,
    is_neox_style,
    query_out,
    key_out,
    device,
    implementation_index=0,
    pre_gathered=False,
):
    if device == "npu":
        infini.ops.rotary_embedding(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox_style,
            rotary_dim,
            query_out,
            key_out,
            pre_gathered,
            implementation_index=implementation_index,
            stream=get_stream(query.device),
        )
    else:
        infini.ops.rotary_embedding(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            is_neox_style,
            rotary_dim,
            query_out,
            key_out,
            pre_gathered,
        )

    return query_out, key_out


def _ref_rotary_embedding(
    positions, query, key, cos_sin_cache, head_size, rotary_dim, is_neox_style
):
    """PyTorch reference for RoPE.

    ``cos_sin_cache`` layout: ``[max_seq_len, rotary_dim]`` where the first
    ``rotary_dim // 2`` columns are cos and the rest are sin.

    Accepts both 2D ``[T, N*D]`` and 3D ``[T, N, D]`` inputs.  When ``key``
    is ``None`` only the query is rotated (MLA).
    """
    T = query.size(0)
    R = rotary_dim
    half_R = R // 2

    # Reshape to 3D for computation if input is 2D.
    q_is_2d = query.ndim == 2
    q3d = query.view(T, -1, head_size) if q_is_2d else query
    k3d = None

    if key is not None:
        k3d = key.view(T, -1, head_size) if q_is_2d else key

    cos_sin = cos_sin_cache.float()
    cos_half = cos_sin[:, :half_R]
    sin_half = cos_sin[:, half_R:]

    def apply_rope(x):
        out = x.float().clone()

        for t in range(T):
            p = positions[t].item()
            c = cos_half[p]
            s = sin_half[p]

            if is_neox_style:
                x1 = x[t, :, :half_R].float()
                x2 = x[t, :, half_R:R].float()
                out[t, :, :half_R] = c * x1 - s * x2
                out[t, :, half_R:R] = c * x2 + s * x1
            else:
                # GPT-J interleave: only the first `rotary_dim` features
                # rotate, and within them even/odd indices form the pairs.
                x1 = x[t, :, 0:R:2].float()
                x2 = x[t, :, 1:R:2].float()
                out[t, :, 0:R:2] = c * x1 - s * x2
                out[t, :, 1:R:2] = c * x2 + s * x1

        return out.to(x.dtype)

    ref_q = apply_rope(q3d)
    ref_k = apply_rope(k3d) if k3d is not None else None

    # Flatten back to 2D if input was 2D.
    if q_is_2d:
        ref_q = ref_q.view(T, -1)

        if ref_k is not None:
            ref_k = ref_k.view(T, -1)

    return ref_q, ref_k


def _assert_close(actual, expected, rtol, atol):
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Max diff: {(actual.float() - expected.float()).abs().max().item()}"
    )


@pytest.mark.parametrize(
    "num_heads, head_size",
    (
        (32, 128),
        (8, 64),
    ),
)
@pytest.mark.parametrize("is_neox_style", (True, False))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_rotary_embedding_full(
    num_heads,
    head_size,
    is_neox_style,
    implementation_index,
    dtype,
    rtol,
    atol,
    device,
):
    """Full rotary: ``rotary_dim == head_size``."""
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    # Only implementation 0 (`aclnnApplyRotaryPosEmbV2`) is still limited to
    # `rotaryMode="half"`; implementation 1 (ATB `RopeParam`) plumbs
    # `rotaryCoeff=head_size` for the non-neox (interleave) case.
    if device == "npu" and not is_neox_style and implementation_index == 0:
        pytest.skip(
            'Ascend `aclnnApplyRotaryPosEmbV2` only supports `rotaryMode="half"`'
        )

    # `aclnnApplyRotaryPosEmbV2` accumulates with ~4 ULP error for float16.
    if device == "npu" and dtype == torch.float16:
        atol = 0.01

    num_kv_heads = num_heads
    rotary_dim = head_size
    num_tokens = 16
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

    q_out, k_out = _rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        rotary_dim,
        is_neox_style,
        query_out,
        key_out,
        device,
        implementation_index=implementation_index,
    )

    ref_q, ref_k = _ref_rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        rotary_dim,
        is_neox_style,
    )

    _assert_close(q_out, ref_q, rtol, atol)
    _assert_close(k_out, ref_k, rtol, atol)


def _rotary_embedding_atb(
    positions,
    query,
    key,
    cos_sin_cache,
    head_size,
    rotary_dim,
    is_neox_style,
    query_out,
    key_out,
):
    """Call rotary embedding with ATB implementation (index=1)."""
    infini.ops.rotary_embedding(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox_style,
        rotary_dim,
        query_out,
        key_out,
        implementation_index=1,
        stream=get_stream(query.device),
    )

    return query_out, key_out


@pytest.mark.parametrize("num_tokens", (1, 4, 16))
@pytest.mark.parametrize(
    "num_heads, head_size",
    (
        (32, 128),
        (8, 64),
    ),
)
@pytest.mark.parametrize("device", ("npu",))
def test_rotary_embedding_atb(num_tokens, num_heads, head_size, device):
    """ATB `RopeParam` path (implementation_index=1), fp16 only."""
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    active_indices = infini.ops.RotaryEmbedding.active_implementation_indices(device)

    if 1 not in active_indices:
        pytest.skip("ATB implementation (index=1) not active on this build")

    dtype = torch.float16
    rtol = 1e-3
    atol = 0.01
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

    q_out, k_out = _rotary_embedding_atb(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        rotary_dim,
        True,
        query_out,
        key_out,
    )

    ref_q, ref_k = _ref_rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        rotary_dim,
        True,
    )

    _assert_close(q_out, ref_q, rtol, atol)
    _assert_close(k_out, ref_k, rtol, atol)


@pytest.mark.parametrize("num_tokens", (1, 4, 16))
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
        (torch.float16, 1e-3, 0.01),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_rotary_embedding_2d(
    num_tokens, num_heads, head_size, implementation_index, dtype, rtol, atol, device
):
    """2D ``[T, N*D]`` layout (vLLM convention) for both CANN and ATB paths."""
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

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

    # 2D layout: [T, N*D].
    query = randn_strided(
        (num_tokens, num_heads * head_size),
        None,
        dtype=dtype,
        device=device,
    )
    key = randn_strided(
        (num_tokens, num_kv_heads * head_size),
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

    if device == "npu":
        infini.ops.rotary_embedding(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            True,
            rotary_dim,
            query_out,
            key_out,
            implementation_index=implementation_index,
            stream=get_stream(query.device),
        )
    else:
        infini.ops.rotary_embedding(
            positions,
            query,
            key,
            head_size,
            cos_sin_cache,
            True,
            rotary_dim,
            query_out,
            key_out,
            implementation_index=implementation_index,
        )

    ref_q, ref_k = _ref_rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        rotary_dim,
        True,
    )

    _assert_close(query_out, ref_q, rtol, atol)
    _assert_close(key_out, ref_k, rtol, atol)


@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_size, rotary_dim",
    (
        (32, 8, 128, 64),
        (16, 4, 64, 32),
    ),
)
@pytest.mark.parametrize("is_neox_style", (True, False))
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
@pytest.mark.parametrize("device", ("npu",))
def test_rotary_embedding_partial(
    num_heads,
    num_kv_heads,
    head_size,
    rotary_dim,
    is_neox_style,
    dtype,
    rtol,
    atol,
    device,
):
    """Partial rotary: ``rotary_dim < head_size`` via implementation_index=2.

    Only `aclnnRopeWithSinCosCache` (impl=2) supports partial rotary among
    the Ascend fused APIs — V2 (impl=0) and ATB `RopeParam` (impl=1) both
    require `cos.D == sin.D == x.D`.  Covers both neox and GPT-J styles.
    """
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    if device == "npu":
        active_indices = infini.ops.RotaryEmbedding.active_implementation_indices(
            device
        )

        if 2 not in active_indices:
            pytest.skip(
                "`aclnnRopeWithSinCosCache` (implementation_index=2) not "
                "active on this build; it is the only Ascend fused API "
                "that supports partial rotary (`rotary_dim < head_size`)."
            )

    num_tokens = 16
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

    q_out, k_out = _rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        rotary_dim,
        is_neox_style,
        query_out,
        key_out,
        device,
        implementation_index=2,
    )

    ref_q, ref_k = _ref_rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        rotary_dim,
        is_neox_style,
    )

    _assert_close(q_out, ref_q, rtol, atol)
    _assert_close(k_out, ref_k, rtol, atol)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        # V2 accumulates ~4 ULP error in fp16 (kernel.h doc: max diff ~0.008);
        # ATB `RopeParam` is similar.  Use atol=5e-3 for honest headroom.
        (torch.float16, 1e-2, 5e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
def test_rotary_embedding_inplace(implementation_index, dtype, rtol, atol, device):
    """Verify the inplace path (`query_out` / `key_out` omitted).

    Matches vLLM's `RotaryEmbedding.forward(positions, query, key)`
    convention where the op mutates `query` / `key` directly.
    """
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    num_tokens = 4
    num_heads = 8
    num_kv_heads = 8
    head_size = 64
    rotary_dim = head_size
    max_seq_len = 32

    positions = randint_strided(
        0, max_seq_len, (num_tokens,), None, dtype=torch.int64, device=device
    )
    query = randn_strided(
        (num_tokens, num_heads, head_size), None, dtype=dtype, device=device
    )
    key = randn_strided(
        (num_tokens, num_kv_heads, head_size), None, dtype=dtype, device=device
    )
    cos_sin_cache = randn_strided(
        (max_seq_len, rotary_dim), None, dtype=dtype, device=device
    )

    # Reference: apply RoPE to clones of the original inputs.
    ref_q, ref_k = _ref_rotary_embedding(
        positions,
        query.clone(),
        key.clone(),
        cos_sin_cache,
        head_size,
        rotary_dim,
        is_neox_style=True,
    )

    # Inplace call — no `query_out` / `key_out` supplied.
    infini.ops.rotary_embedding(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        True,
        rotary_dim,
        implementation_index=implementation_index,
        stream=get_stream(query.device),
    )

    _assert_close(query, ref_q, rtol, atol)
    _assert_close(key, ref_k, rtol, atol)

