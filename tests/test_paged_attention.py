import infini.ops
import pytest
import torch

from tests.utils import Payload, get_npu_stream, randn_strided


def _atb_pa_available():
    """Check whether ATB PagedAttention works on the current hardware.

    ATB PA is known to crash during `Setup` on Ascend 910B (CANN 8.5.x).
    Returns True only when a minimal smoke call succeeds.
    """
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return False

    if not infini.ops.PagedAttention.active_implementation_indices("ascend"):
        return False

    try:
        B, N, Nkv, D, bs = 1, 4, 4, 64, 16
        q = torch.randn(B, N, D, dtype=torch.float16, device="npu")
        kc = torch.randn(1, bs, Nkv, D, dtype=torch.float16, device="npu")
        vc = torch.randn(1, bs, Nkv, D, dtype=torch.float16, device="npu")
        bt = torch.zeros(B, 1, dtype=torch.int32, device="npu")
        sl = torch.tensor([bs], dtype=torch.int32, device="npu")
        o = torch.zeros(B, N, D, dtype=torch.float16, device="npu")
        infini.ops.paged_attention(
            q,
            kc,
            vc,
            sl,
            bt,
            N,
            Nkv,
            D,
            1.0 / D**0.5,
            bs,
            o,
            stream=get_npu_stream(q),
        )
        torch.npu.synchronize()

        return True
    except Exception:
        return False


_skip_no_atb_pa = pytest.mark.skipif(
    not _atb_pa_available(),
    reason="ATB PagedAttention not supported on this hardware",
)


@_skip_no_atb_pa
@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_size, block_size",
    (
        (32, 8, 128, 128),
        (16, 4, 64, 128),
        (32, 32, 128, 128),  # MHA
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
@pytest.mark.parametrize("device", ("npu",))
def test_paged_attention_basic(
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    dtype,
    rtol,
    atol,
    device,
):
    """Basic paged decode attention with contiguous block assignments."""
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    num_reqs = 4
    kv_len = 16
    num_blocks_per_req = (kv_len + block_size - 1) // block_size
    num_blocks = num_reqs * num_blocks_per_req
    scale = 1.0 / head_size**0.5

    query = randn_strided(
        (num_reqs, num_heads, head_size), None, dtype=dtype, device=device
    )
    key_cache = randn_strided(
        (num_blocks, block_size, num_kv_heads, head_size),
        None,
        dtype=dtype,
        device=device,
    )
    value_cache = randn_strided(
        (num_blocks, block_size, num_kv_heads, head_size),
        None,
        dtype=dtype,
        device=device,
    )
    output = torch.empty((num_reqs, num_heads, head_size), dtype=dtype, device=device)

    # Block table: request i uses blocks [i*num_blocks_per_req, ...].
    block_table = torch.zeros(
        (num_reqs, num_blocks_per_req), dtype=torch.int32, device=device
    )

    for i in range(num_reqs):
        for j in range(num_blocks_per_req):
            block_table[i, j] = i * num_blocks_per_req + j

    # Context lengths (total KV length per request).
    seq_lens = torch.full((num_reqs,), kv_len, dtype=torch.int32, device=device)

    return Payload(
        lambda q, kc, vc, sl, bt, o: _paged_attention(
            q,
            kc,
            vc,
            sl,
            bt,
            num_heads,
            num_kv_heads,
            head_size,
            scale,
            block_size,
            o,
        ),
        lambda q, kc, vc, sl, bt, o: _ref_paged_attention(
            q,
            kc,
            vc,
            sl,
            bt,
            num_heads,
            num_kv_heads,
            head_size,
            scale,
            block_size,
        ),
        (query, key_cache, value_cache, seq_lens, block_table, output),
        {},
        rtol=rtol,
        atol=atol,
    )


@_skip_no_atb_pa
@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_size, block_size",
    ((32, 8, 128, 128),),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 5e-3),
    ),
)
@pytest.mark.parametrize("device", ("npu",))
def test_paged_attention_variable_seq_lens(
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    dtype,
    rtol,
    atol,
    device,
):
    """Paged decode attention where each request has a different KV length."""
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    kv_lens = [8, 32, 16, 128]
    num_reqs = len(kv_lens)
    max_blocks_per_req = max((kv + block_size - 1) // block_size for kv in kv_lens)
    num_blocks = sum((kv + block_size - 1) // block_size for kv in kv_lens)
    scale = 1.0 / head_size**0.5

    query = randn_strided(
        (num_reqs, num_heads, head_size), None, dtype=dtype, device=device
    )
    key_cache = randn_strided(
        (num_blocks, block_size, num_kv_heads, head_size),
        None,
        dtype=dtype,
        device=device,
    )
    value_cache = randn_strided(
        (num_blocks, block_size, num_kv_heads, head_size),
        None,
        dtype=dtype,
        device=device,
    )
    output = torch.empty((num_reqs, num_heads, head_size), dtype=dtype, device=device)

    # Block table: assign blocks sequentially.
    block_table = torch.zeros(
        (num_reqs, max_blocks_per_req), dtype=torch.int32, device=device
    )
    block_idx = 0

    for i in range(num_reqs):
        n_blocks = (kv_lens[i] + block_size - 1) // block_size

        for j in range(n_blocks):
            block_table[i, j] = block_idx
            block_idx += 1

    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    return Payload(
        lambda q, kc, vc, sl, bt, o: _paged_attention(
            q,
            kc,
            vc,
            sl,
            bt,
            num_heads,
            num_kv_heads,
            head_size,
            scale,
            block_size,
            o,
        ),
        lambda q, kc, vc, sl, bt, o: _ref_paged_attention(
            q,
            kc,
            vc,
            sl,
            bt,
            num_heads,
            num_kv_heads,
            head_size,
            scale,
            block_size,
        ),
        (query, key_cache, value_cache, seq_lens, block_table, output),
        {},
        rtol=rtol,
        atol=atol,
    )


@_skip_no_atb_pa
@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_size, block_size",
    ((32, 8, 128, 128),),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    ((torch.float16, 1e-3, 1e-3),),
)
@pytest.mark.parametrize("device", ("npu",))
def test_paged_attention_single_request(
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    dtype,
    rtol,
    atol,
    device,
):
    """Single request decode (batch_size=1)."""
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    num_reqs = 1
    kv_len = 64
    num_blocks_per_req = (kv_len + block_size - 1) // block_size
    num_blocks = num_blocks_per_req
    scale = 1.0 / head_size**0.5

    query = randn_strided(
        (num_reqs, num_heads, head_size), None, dtype=dtype, device=device
    )
    key_cache = randn_strided(
        (num_blocks, block_size, num_kv_heads, head_size),
        None,
        dtype=dtype,
        device=device,
    )
    value_cache = randn_strided(
        (num_blocks, block_size, num_kv_heads, head_size),
        None,
        dtype=dtype,
        device=device,
    )
    output = torch.empty((num_reqs, num_heads, head_size), dtype=dtype, device=device)

    block_table = torch.arange(
        num_blocks_per_req, dtype=torch.int32, device=device
    ).unsqueeze(0)

    seq_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)

    return Payload(
        lambda q, kc, vc, sl, bt, o: _paged_attention(
            q,
            kc,
            vc,
            sl,
            bt,
            num_heads,
            num_kv_heads,
            head_size,
            scale,
            block_size,
            o,
        ),
        lambda q, kc, vc, sl, bt, o: _ref_paged_attention(
            q,
            kc,
            vc,
            sl,
            bt,
            num_heads,
            num_kv_heads,
            head_size,
            scale,
            block_size,
        ),
        (query, key_cache, value_cache, seq_lens, block_table, output),
        {},
        rtol=rtol,
        atol=atol,
    )


@_skip_no_atb_pa
@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "num_heads, num_kv_heads, head_size, block_size",
    ((32, 8, 128, 128),),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    ((torch.float16, 1e-3, 1e-3),),
)
@pytest.mark.parametrize("device", ("npu",))
def test_paged_attention_host_tensors(
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    dtype,
    rtol,
    atol,
    device,
):
    """Paged decode with caller-provided host tensors (D2H-free path)."""
    if device == "npu" and not (hasattr(torch, "npu") and torch.npu.is_available()):
        pytest.skip("NPU not available")

    num_reqs = 4
    kv_len = 16
    num_blocks_per_req = (kv_len + block_size - 1) // block_size
    num_blocks = num_reqs * num_blocks_per_req
    scale = 1.0 / head_size**0.5

    query = randn_strided(
        (num_reqs, num_heads, head_size), None, dtype=dtype, device=device
    )
    key_cache = randn_strided(
        (num_blocks, block_size, num_kv_heads, head_size),
        None,
        dtype=dtype,
        device=device,
    )
    value_cache = randn_strided(
        (num_blocks, block_size, num_kv_heads, head_size),
        None,
        dtype=dtype,
        device=device,
    )
    output = torch.empty(
        (num_reqs, num_heads, head_size), dtype=dtype, device=device
    )

    block_table = torch.zeros(
        (num_reqs, num_blocks_per_req), dtype=torch.int32, device=device
    )

    for i in range(num_reqs):
        for j in range(num_blocks_per_req):
            block_table[i, j] = i * num_blocks_per_req + j

    seq_lens = torch.full(
        (num_reqs,), kv_len, dtype=torch.int32, device=device
    )

    # CPU copies for the D2H-free path.
    seq_lens_cpu = seq_lens.cpu().contiguous()
    block_table_cpu = block_table.cpu().contiguous()

    return Payload(
        lambda q, kc, vc, sl, bt, o: _paged_attention_with_host(
            q,
            kc,
            vc,
            sl,
            bt,
            num_heads,
            num_kv_heads,
            head_size,
            scale,
            block_size,
            o,
            seq_lens_cpu,
            block_table_cpu,
        ),
        lambda q, kc, vc, sl, bt, o: _ref_paged_attention(
            q,
            kc,
            vc,
            sl,
            bt,
            num_heads,
            num_kv_heads,
            head_size,
            scale,
            block_size,
        ),
        (query, key_cache, value_cache, seq_lens, block_table, output),
        {},
        rtol=rtol,
        atol=atol,
    )


def _paged_attention_with_host(
    query,
    key_cache,
    value_cache,
    seq_lens,
    block_table,
    num_heads,
    num_kv_heads,
    head_size,
    scale,
    block_size,
    output,
    seq_lens_host,
    block_table_host,
):
    """Call paged attention with caller-provided host tensors."""
    infini.ops.paged_attention(
        query,
        key_cache,
        value_cache,
        seq_lens,
        block_table,
        num_heads,
        num_kv_heads,
        head_size,
        scale,
        block_size,
        output,
        seq_lens_host=seq_lens_host,
        block_table_host=block_table_host,
        stream=get_npu_stream(query),
    )

    return output


def _paged_attention(
    query,
    key_cache,
    value_cache,
    seq_lens,
    block_table,
    num_heads,
    num_kv_heads,
    head_size,
    scale,
    block_size,
    output,
):
    if query.device.type == "npu":
        infini.ops.paged_attention(
            query,
            key_cache,
            value_cache,
            seq_lens,
            block_table,
            num_heads,
            num_kv_heads,
            head_size,
            scale,
            block_size,
            output,
            stream=get_npu_stream(query),
        )
    else:
        infini.ops.paged_attention(
            query,
            key_cache,
            value_cache,
            seq_lens,
            block_table,
            num_heads,
            num_kv_heads,
            head_size,
            scale,
            block_size,
            output,
        )

    return output


def _ref_paged_attention(
    query,
    key_cache,
    value_cache,
    seq_lens,
    block_table,
    num_heads,
    num_kv_heads,
    head_size,
    scale,
    block_size,
):
    """PyTorch SDPA reference for paged decode attention."""
    sl = seq_lens.cpu()
    bt = block_table.cpu()
    kc = key_cache.cpu().float()
    vc = value_cache.cpu().float()
    q_cpu = query.cpu().float()
    num_reqs = bt.size(0)
    outputs = []

    for i in range(num_reqs):
        q = q_cpu[i : i + 1]  # [1, N, D]
        kv_len = int(sl[i].item())

        # Gather K and V from paged cache.
        # Cache layout: [num_blocks, block_size, Nkv, D].
        blocks = bt[i]
        k_pages = []
        v_pages = []
        remaining = kv_len

        for b in blocks:
            if remaining <= 0:
                break

            take = min(remaining, block_size)
            k_pages.append(kc[int(b.item()), :take, :, :])
            v_pages.append(vc[int(b.item()), :take, :, :])
            remaining -= take

        # [kv_len, Nkv, D]
        k = torch.cat(k_pages, dim=0)
        v = torch.cat(v_pages, dim=0)

        # SDPA reference with GQA expansion.
        # q: [1, N, D] -> [N, 1, D]
        q_t = q.transpose(0, 1)
        # k, v: [kv_len, Nkv, D] -> [Nkv, kv_len, D]
        k_t = k.transpose(0, 1)
        v_t = v.transpose(0, 1)

        if num_kv_heads < num_heads:
            ratio = num_heads // num_kv_heads
            k_t = k_t.repeat_interleave(ratio, dim=0)
            v_t = v_t.repeat_interleave(ratio, dim=0)

        # [N, 1, D] and [N, kv_len, D] -> [1, N, 1, D] and [1, N, kv_len, D]
        q_4d = q_t.unsqueeze(0)
        k_4d = k_t.unsqueeze(0)
        v_4d = v_t.unsqueeze(0)

        # Decode: query attends to all past KV (no causal mask).
        out = torch.nn.functional.scaled_dot_product_attention(
            q_4d,
            k_4d,
            v_4d,
            scale=scale,
            is_causal=False,
        )

        # [1, N, 1, D] -> [1, N, D]
        outputs.append(out.squeeze(0).transpose(0, 1).squeeze(0).unsqueeze(0))

    return torch.cat(outputs, dim=0).to(query.dtype).to(query.device)
