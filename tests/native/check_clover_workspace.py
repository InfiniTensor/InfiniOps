import infini.ops as ops
import torch
import workspace_ops

torch.manual_seed(29)
s = torch.cuda.current_stream().cuda_stream
d = "cuda"
dtype = torch.bfloat16
for n, k in ((1, 8), (32, 8), (64, 8), (32, 3)):
    e, h, i, bm = 16, 128, 64, 16
    x = torch.randn(n, h, device=d, dtype=dtype) * 0.2
    w13 = torch.randn(e, 2 * i, h, device=d, dtype=dtype) * 0.1
    w2 = torch.randn(e, h, i, device=d, dtype=dtype) * 0.1
    scores = torch.randn(n, e, device=d).softmax(-1)
    weights, ids = scores.topk(k, dim=-1)
    weights = weights.contiguous()
    ids = ids.int().contiguous()
    cap = max(n * k * bm if n * k < e else n * k + e * (bm - 1), bm)
    sorted_ids = torch.empty(cap, device=d, dtype=torch.int32)
    experts = torch.empty((cap + bm - 1) // bm, device=d, dtype=torch.int32)
    total = torch.empty(1, device=d, dtype=torch.int32)
    ops.moe_align_block_size(ids, None, e, bm, sorted_ids, experts, total, stream=s)
    y = torch.full_like(x, float("nan"))
    scratch = torch.empty(n * k * (2 * i + h) * 2, device=d, dtype=torch.uint8)
    workspace_ops.moe(
        [y, x, w13, w2, weights, ids, sorted_ids, experts, total], bm, scratch, s
    )
    ref = torch.zeros_like(x, dtype=torch.float32)
    for token in range(n):
        for r in range(k):
            expert = int(ids[token, r])
            mid = x[token] @ w13[expert].T
            act = (torch.nn.functional.silu(mid[:i].float()) * mid[i:].float()).to(
                dtype
            )
            value = act @ w2[expert].T
            ref[token] += value.float() * weights[token, r]
    torch.testing.assert_close(y, ref.to(dtype), rtol=0.025, atol=0.003)
    print("PASS MoE", n, float((y - ref).abs().max()), flush=True)
q = torch.randn(1, 11, 2, 128, device=d, dtype=dtype)
k = torch.randn_like(q)
v = torch.randn(1, 11, 4, 128, device=d, dtype=dtype)
g = -torch.rand(1, 11, 4, device=d)
beta = torch.rand_like(g)
state = torch.randn(3, 4, 128, 128, device=d)
old = state.clone()
out = torch.empty_like(v)
offsets = torch.tensor([0, 4, 11], device=d, dtype=torch.int32)
init = torch.tensor([0, 1], device=d, dtype=torch.int32)
final = init.clone()
ops.clover_recurrent_gdn(q, k, v, g, beta, offsets, init, final, state, out, stream=s)
qn = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
kn = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
qn = qn.repeat_interleave(2, 2) / 128**0.5
kn = kn.repeat_interleave(2, 2)
ref = torch.empty_like(v)
for b, (start, end) in enumerate(((0, 4), (4, 11))):
    st = old[b].clone()
    for t in range(start, end):
        st = st * g[0, t].exp()[:, None, None]
        remembered = (st * kn[0, t, :, None, :]).sum(-1)
        delta = (v[0, t].float() - remembered) * beta[0, t, :, None]
        st += delta[:, :, None] * kn[0, t, :, None, :]
        ref[0, t] = (st * qn[0, t, :, None, :]).sum(-1)
    torch.testing.assert_close(state[b], st, rtol=0.001, atol=0.001)
torch.testing.assert_close(out, ref, rtol=0.02, atol=0.01)
assert torch.equal(state[2], old[2])
print("PASS recurrent GDN", float((out - ref).abs().max()), flush=True)
