"""Correctness tests for the native CloverLocalMemory provider.

Golden model replicates the Triton reference semantics: L2-normalized q/k,
block-local softmax with the gate-decayed clean-state history bucket
(history logit = bias + c * log(valid_count / block_size)), per-head
scalars, dead slots and per-row validity. Cross-checked against Triton.
"""

import infini.ops
import pytest
import torch

if not hasattr(infini.ops, "CloverLocalMemory"):
    pytest.skip("`CloverLocalMemory` is not available", allow_module_level=True)

NATIVE, TRITON = 0, 10


def golden(q, k, v, g, state, indices, valid, ls, hb, hc, block, scale=1 / 128**0.5):
    _, tokens, nqh, kd = q.shape
    _, _, nvh, vd = v.shape
    nkh = k.shape[2]
    requests = indices.numel()
    out = torch.zeros_like(v, dtype=torch.float32)
    qf, kf, vf = q.float(), k.float(), v.float()
    for r in range(requests):
        slot = int(indices[r])
        ra = 0 <= slot < state.size(0)
        ss = max(slot, 0)
        for h in range(nvh):
            qh = h // (nvh // nqh)
            kh = h // (nvh // nkh)
            blk_q = qf[0, r * block:(r + 1) * block, qh]           # [T,K]
            blk_k = kf[0, r * block:(r + 1) * block, kh]           # [T,K]
            blk_v = vf[0, r * block:(r + 1) * block, h]            # [T,V]
            qn = torch.nn.functional.normalize(blk_q, dim=1, eps=1e-6**0.5 * 1e-3)
            qn = blk_q / (blk_q.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
            kn = blk_k / (blk_k.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
            scores = (qn @ kn.T) * float(ls[h])
            local_valid = torch.tensor(
                [1.0 if (ra and int(valid[r * block + j]) != 0) else 0.0
                 for j in range(block)], device=q.device)
            logits = scores.masked_fill(local_valid[None, :] == 0, float("-inf"))
            vbs = max(local_valid.sum().item(), 1.0)
            bias = float(hb[h]) + float(hc[h]) * torch.log(
                torch.tensor(vbs / block, device=q.device))
            hist = qn @ state[ss, h].float()                        # [T,V]
            row_max = torch.maximum(logits.max(-1).values,
                                    torch.full_like(logits.max(-1).values, bias))
            el = torch.exp(logits - row_max[:, None])
            eh = torch.exp(bias - row_max)
            denom = el.sum(-1) + eh
            w = el / denom[:, None]
            hw = eh / denom
            gate = scale * torch.exp(g[0, r * block:(r + 1) * block, h].float())
            local = w @ blk_v
            o = hist * (gate * hw)[:, None] + local
            active = torch.tensor(
                [1.0 if (ra and int(valid[r * block + t]) != 0) else 0.0
                 for t in range(block)], device=q.device)
            out[0, r * block:(r + 1) * block, h] = o * active[:, None]
    return out


@pytest.mark.parametrize("block", (8, 32))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("heads", ((4, 4, 8), (2, 2, 2)))  # (nq, nk, nv)
def test_local_memory(device, block, dtype, heads):
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    impls = infini.ops.CloverLocalMemory.active_implementation_indices("nvidia")
    if NATIVE not in impls or TRITON not in impls:
        pytest.skip(f"requires impls 0 and 10, active={impls}")
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(41)
    requests, slots, dim = 3, 4, 128
    nqh, nkh, nvh = heads
    tokens = requests * block

    q = torch.randn(1, tokens, nqh, dim, device=device, dtype=dtype) * 0.5
    k = torch.randn(1, tokens, nkh, dim, device=device, dtype=dtype) * 0.5
    v = torch.randn(1, tokens, nvh, dim, device=device, dtype=dtype) * 0.5
    g = torch.randn(1, tokens, nvh, device=device, dtype=torch.float32) * 0.1 - 1.0
    state = torch.randn(slots, nvh, dim, dim, device=device, dtype=torch.float32) * 0.1
    indices = torch.tensor([0, 1, -1], device=device, dtype=torch.int32)
    valid = torch.ones(tokens, device=device, dtype=torch.int8)
    valid[::4] = 0  # scattered inactive rows
    ls = torch.randn(nvh, device=device, dtype=dtype) * 0.3 + 1.0
    hb = torch.randn(nvh, device=device, dtype=dtype) * 0.2
    hc = torch.randn(nvh, device=device, dtype=dtype) * 0.1
    out_n = torch.empty_like(v)
    out_t = torch.empty_like(v)

    infini.ops.clover_local_memory(q, k, v, g, state, indices, valid, ls, hb, hc,
                                   block, out_n, stream=stream,
                                   implementation_index=NATIVE)
    infini.ops.clover_local_memory(q, k, v, g, state, indices, valid, ls, hb, hc,
                                   block, out_t, stream=stream,
                                   implementation_index=TRITON)
    torch.cuda.synchronize()

    ref = golden(q, k, v, g, state, indices, valid, ls, hb, hc, block).to(v.device)
    scale = max(ref.abs().max().item(), 1.0)
    torch.testing.assert_close(out_n.float(), ref, rtol=3e-2, atol=scale * 3e-2)
    torch.testing.assert_close(out_n.float(), out_t.float(), rtol=3e-2,
                               atol=scale * 3e-2)
