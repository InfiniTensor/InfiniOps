"""Correctness tests for the native CloverLocalMemory providers.

Golden model replicates the Triton reference semantics: L2-normalized q/k
rounded to the element type, block-local softmax with the gate-decayed
clean-state history bucket (history logit = bias + c * log(valid_count / 32),
the reference sizes it against a full 32-token block), per-head scalars, dead
slots and per-row validity. The state has unit magnitude, so the history term
is a fair share of the output and a broken history cannot hide in the
tolerance. Cross-checked against Triton, including bit level.
"""

import math

import infini.ops
import pytest
import torch

if not hasattr(infini.ops, "CloverLocalMemory"):
    pytest.skip("`CloverLocalMemory` is not available", allow_module_level=True)

NATIVE, TRITON = 0, 10
REFERENCE_BLOCK = 32.0


def golden(q, k, v, g, state, indices, valid, ls, hb, hc, block):
    _, tokens, nqh, kd = q.shape
    _, _, nvh, vd = v.shape
    nkh = k.shape[2]
    dtype = q.dtype
    reduced = dtype in (torch.bfloat16, torch.float16)
    out = torch.zeros_like(v, dtype=torch.float32)
    qf, kf, vf = q.float(), k.float(), v.float()
    for r in range(indices.numel()):
        slot = int(indices[r])
        ra = 0 <= slot < state.size(0)
        if not ra:
            continue  # a dead slot leaves the zeros of the reference
        for h in range(nvh):
            qh = h // (nvh // nqh)
            kh = h // (nvh // nkh)
            blk_q = qf[0, r * block:(r + 1) * block, qh]           # [T,K]
            blk_k = kf[0, r * block:(r + 1) * block, kh]           # [T,K]
            blk_v = vf[0, r * block:(r + 1) * block, h]            # [T,V]
            qn = blk_q / (blk_q.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
            kn = blk_k / (blk_k.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
            key_valid = torch.tensor(
                [int(valid[r * block + j]) != 0 for j in range(block)], device=q.device)
            row_valid = torch.tensor(
                [int(valid[r * block + t]) != 0 for t in range(block)], device=q.device)
            if reduced:  # the reference rounds these to the element type
                qn, kn = qn.to(dtype).float(), kn.to(dtype).float()
                st = state[slot, h].to(dtype).float()
            else:
                st = state[slot, h].float()
            scores = (qn @ kn.T) * float(ls[h])
            logits = scores.masked_fill(~key_valid[None, :], float("-inf"))
            bias = float(hb[h]) + float(hc[h]) * math.log(
                max(int(key_valid.sum()), 1) / REFERENCE_BLOCK)
            hist = qn @ st.T                                       # [T,V]; state is [V, K]
            row_max = logits.max(-1).values.clamp_min(bias)
            el = torch.exp(logits - row_max[:, None])
            eh = torch.exp(bias - row_max)
            denom = el.sum(-1) + eh
            w = el / denom[:, None]
            hw = eh / denom
            if reduced:
                w = w.to(dtype).float()
            gate = 128**-0.5 * torch.exp(g[0, r * block:(r + 1) * block, h].float())
            o = hist * (gate * hw)[:, None] + w @ blk_v
            out[0, r * block:(r + 1) * block, h] = o * row_valid[:, None].to(o.dtype)
    return out


def inputs(requests, block, heads, dtype, slots=4, index_dtype=torch.int32):
    nqh, nkh, nvh = heads
    tokens = requests * block
    q = torch.randn(1, tokens, nqh, 128, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(1, tokens, nkh, 128, device="cuda", dtype=dtype) * 0.5
    v = torch.randn(1, tokens, nvh, 128, device="cuda", dtype=dtype) * 0.5
    g = torch.randn(1, tokens, nvh, device="cuda", dtype=torch.float32) * 0.5 - 0.3
    # Unit magnitude: the history term is a fair share of the output.
    state = torch.randn(slots, nvh, 128, 128, device="cuda", dtype=torch.float32)
    indices = torch.tensor([(r % slots) if r % 4 != 3 else -1
                            for r in range(requests)], device="cuda").to(index_dtype)
    valid = torch.ones(tokens, device="cuda", dtype=torch.int8)
    valid[::5] = 0  # scattered inactive rows and keys
    ls = torch.randn(nvh, device="cuda", dtype=dtype) * 0.3 + 1.0
    hb = torch.randn(nvh, device="cuda", dtype=dtype) * 0.2
    hc = torch.randn(nvh, device="cuda", dtype=dtype) * 0.1
    return q, k, v, g, state, indices, valid, ls, hb, hc


def run(args, block, impl, out=None):
    q, k, v = args[:3]
    out = torch.empty_like(v) if out is None else out
    infini.ops.clover_local_memory(*args, block, out,
                                   stream=torch.cuda.current_stream().cuda_stream,
                                   implementation_index=impl)
    torch.cuda.synchronize()
    return out


def require_impls(device):
    impls = infini.ops.CloverLocalMemory.active_implementation_indices(device)
    if NATIVE not in impls or TRITON not in impls:
        pytest.skip(f"requires impls 0 and 10, active={impls}")


@pytest.mark.parametrize("block", (8, 16, 32))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("heads", ((4, 4, 8), (8, 8, 16), (2, 2, 2)))  # (nq, nk, nv)
def test_local_memory(device, block, dtype, heads):
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    require_impls(device)
    torch.manual_seed(41)
    args = inputs(3, block, heads, dtype)
    out_n = run(args, block, NATIVE)
    out_t = run(args, block, TRITON)

    ref = golden(*args, block).cuda()
    scale = max(ref.abs().max().item(), 1.0)
    # One unit of the element type at the largest magnitude: with the same
    # rounding points as the reference, only fp32 summation order is left.
    tol = scale * torch.finfo(dtype).eps
    torch.testing.assert_close(out_n.float(), ref, rtol=0, atol=tol)
    torch.testing.assert_close(out_n.float(), out_t.float(), rtol=0, atol=tol)


@pytest.mark.parametrize("requests", (1, 2, 4, 8, 13))
@pytest.mark.parametrize("index_dtype", (torch.int32, torch.int64))
def test_local_memory_batch(device, requests, index_dtype):
    """Every tiling the NVIDIA launcher picks, and out-of-range slots."""
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    require_impls(device)
    torch.manual_seed(43)
    args = list(inputs(requests, 32, (8, 8, 16), torch.bfloat16,
                       index_dtype=index_dtype))
    args[5][-1] = 99  # beyond the pool: inactive like a negative slot
    args = tuple(args)
    out_n = run(args, 32, NATIVE)
    out_t = run(args, 32, TRITON)
    ref = golden(*args, 32).cuda()
    scale = max(ref.abs().max().item(), 1.0)
    tol = scale * torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(out_n.float(), ref, rtol=0, atol=tol)
    torch.testing.assert_close(out_n.float(), out_t.float(), rtol=0, atol=tol)
    # Rows of an inactive request are zeros, as in the reference.
    for r, slot in enumerate(args[5].tolist()):
        if slot < 0 or slot >= args[4].size(0):
            assert torch.equal(out_n[0, r * 32:(r + 1) * 32],
                               torch.zeros_like(out_n[0, r * 32:(r + 1) * 32]))


def test_local_memory_fallback_layouts(device):
    """Layouts the tensor-core kernel does not read take the shared SIMT
    kernel: fp32 scalars, strided rows, a head sharing ratio that is not a
    power of two. All of them must agree with the reference."""
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    require_impls(device)
    torch.manual_seed(47)
    dtype, block, heads = torch.bfloat16, 32, (8, 8, 16)
    cases = {}

    args = list(inputs(2, block, heads, dtype))
    args[7] = args[7].float()  # fp32 per-head scalars
    cases["fp32_scalars"] = tuple(args)

    args = list(inputs(2, block, heads, dtype))
    # q and k rows that are slices of wider tensors: contiguous in the
    # innermost axis but not 16-byte aligned.
    wide = torch.randn(1, 2 * block, 16, 136, device="cuda", dtype=dtype)
    args[0] = wide[..., 4:132]
    args[1] = wide[..., 4:132][:, :, 8:16]
    cases["misaligned_base"] = tuple(args)

    args = list(inputs(2, block, heads, dtype))
    args[0] = torch.randn(1, 2 * block, 8, 256, device="cuda", dtype=dtype)[..., ::2]
    cases["strided_innermost"] = tuple(args)

    args = inputs(2, block, (2, 3, 6), dtype)  # q sharing ratio 3
    cases["odd_head_ratio"] = args

    for name, args in cases.items():
        out_n = run(args, block, NATIVE)
        out_t = run(args, block, TRITON)
        ref = golden(*args, block).cuda()
        scale = max(ref.abs().max().item(), 1.0)
        torch.testing.assert_close(out_n.float(), ref, rtol=3e-2, atol=scale * 3e-2,
                                   msg=lambda m: f"{name}: {m}")
        torch.testing.assert_close(out_n.float(), out_t.float(), rtol=3e-2,
                                   atol=scale * 3e-2)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_local_memory_matches_triton(device, dtype):
    """Bit level: the same MMA and rounding points leave only the
    summation-order noise of the norms and of the softmax sums."""
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    require_impls(device)
    torch.manual_seed(53)
    for requests in (1, 4, 12):
        args = inputs(requests, 32, (8, 8, 16), dtype)
        out_n, out_t = run(args, 32, NATIVE), run(args, 32, TRITON)
        bits_n = out_n.view(torch.int16).long()
        bits_t = out_t.view(torch.int16).long()
        same = (bits_n == bits_t).float().mean().item()
        assert same >= 0.999, f"R={requests}: only {same:.5%} bit-identical"
        diff = (out_n.float() - out_t.float()).abs()
        scale = max(out_t.float().abs().max().item(), 1.0)
        assert diff.max().item() <= scale * 2.0 * torch.finfo(dtype).eps


def test_local_memory_deterministic(device):
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    require_impls(device)
    torch.manual_seed(59)
    for requests in (1, 2, 5):
        args = inputs(requests, 32, (8, 8, 16), torch.bfloat16)
        first = run(args, 32, NATIVE).clone()
        for _ in range(4):
            assert torch.equal(run(args, 32, NATIVE), first), f"R={requests}"


def test_local_memory_reads_state_only(device):
    """The preview reads the clean state and never writes it."""
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    require_impls(device)
    torch.manual_seed(44)
    args = list(inputs(3, 32, (4, 4, 8), torch.bfloat16))
    state0 = args[4].clone()
    run(args, 32, NATIVE)
    run(args, 32, TRITON)
    assert torch.equal(args[4], state0)
