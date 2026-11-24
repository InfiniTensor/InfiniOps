"""Checks the Hygon `moe_fused_dense` against float32 torch references.

On gfx936 the shapes the native kernels accept (block size 16; h and i multiples of 8, at
least 32) take the native path; the remaining ones are delegated to the Triton JIT kernels.
Each path is checked against the reference with its own rounding: the native kernels keep the
gate/up projections in float32 and round the activation once, while the Triton kernels also
round the projections to the element type. Routing follows the model's, top-k weights that
sum to one, spread over the experts or concentrated on a few by the block coreset.

Build the adapter first:

    python tests/native/build_clover_workspace.py

usage: python tests/native/check_moe_fused_dense_hygon.py
"""

import infini.ops as ops
import torch
import workspace_ops

BLOCK = 16
DEV = "cuda"
RTOL, ATOL = 0.025, 0.003
NATIVE, TRITON = 0, 10  # implementation indices
HAS_NATIVE = workspace_ops.moe_has_native_kernels()  # compiled in and not switched off


def uses_native(h, i):
    return HAS_NATIVE and h >= 32 and h % 8 == 0 and i >= 32 and i % 8 == 0


def reference(hidden, w13, w2, weights, ids, round_projections):
    """Float32 throughout, with the activation rounded to the element type once. With
    `round_projections` the gate/up projections are rounded to the element type first, as
    the Triton kernels do. Routes outside [0, experts) contribute nothing."""
    dtype, n, i = hidden.dtype, hidden.shape[0], w2.shape[2]
    topk = ids.shape[1]
    out = torch.zeros(n, hidden.shape[1], device=DEV, dtype=torch.float32)
    flat = ids.view(-1).long()
    tokens = torch.arange(n, device=DEV).repeat_interleave(topk)
    routed = weights.view(-1)
    experts = [e for e in flat.unique().tolist() if 0 <= e < w2.shape[0]]
    for expert in experts:
        sel = (flat == expert).nonzero().squeeze(-1)
        mid = hidden[tokens[sel]].float() @ w13[expert].float().T
        if round_projections:
            mid = mid.to(dtype).float()
        act = (torch.nn.functional.silu(mid[:, :i]) * mid[:, i:]).to(dtype).float()
        out.index_add_(0, tokens[sel], (act @ w2[expert].float().T) * routed[sel, None])
    return out


def make_case(n, k, e, h, i, dtype, hot=0.0, inactive=0):
    """Inputs of one call. With `hot` > 0 that share of the routes goes to 16 experts, which
    gives those experts several blocks; the first `inactive` routes of a token are dropped."""
    scores = -torch.rand(n, e, device=DEV)
    if hot > 0:
        crowded = torch.rand(n, e, device=DEV) < hot
        scores[:, :16] += 2.0 * crowded[:, :16]
    top = scores.topk(k, dim=-1)
    weights = top.values.softmax(-1).float().contiguous()
    ids = top.indices.int().contiguous()
    ids[:, :inactive] = -1
    hidden = torch.randn(n, h, device=DEV, dtype=dtype) * 0.2
    w13 = torch.randn(e, 2 * i, h, device=DEV, dtype=dtype) * 0.1
    w2 = torch.randn(e, h, i, device=DEV, dtype=dtype) * 0.1

    pairs = n * k
    capacity = pairs * BLOCK if pairs < e else pairs + e * (BLOCK - 1)
    sorted_ids = torch.empty(capacity, device=DEV, dtype=torch.int32)
    experts = torch.empty((capacity + BLOCK - 1) // BLOCK, device=DEV, dtype=torch.int32)
    total = torch.empty(1, device=DEV, dtype=torch.int32)
    ops.moe_align_block_size(
        ids, None, e, BLOCK, sorted_ids, experts, total,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    return hidden, w13, w2, weights, ids, sorted_ids, experts, total


def run(hidden, w13, w2, weights, ids, sorted_ids, experts, total, implementation):
    out = torch.full_like(hidden, float("nan"))
    tensors = [out, hidden, w13, w2, weights, ids, sorted_ids, experts, total]
    scratch = torch.empty(
        workspace_ops.moe_workspace_size(tensors, BLOCK), device=DEV, dtype=torch.uint8
    )
    workspace_ops.moe(
        tensors, BLOCK, scratch, torch.cuda.current_stream().cuda_stream, implementation
    )
    torch.cuda.synchronize()
    return out


def max_error(out, ref):
    return float((out.float() - ref).abs().max())


def check(name, n, k, e, h, i, dtype, hot=0.0, inactive=0):
    case = make_case(n, k, e, h, i, dtype, hot, inactive)
    native = uses_native(h, i)
    out = run(*case, NATIVE)
    ref = reference(case[0], case[1], case[2], case[3], case[4], round_projections=not native)
    torch.testing.assert_close(out.float(), ref, rtol=RTOL, atol=ATOL)
    print(
        f"PASS {name}: {'native' if native else 'Triton'} path, n={n} k={k} e={e} h={h} i={i} "
        f"{str(dtype)[6:]}, max|err| {max_error(out, ref):.4f}",
        flush=True,
    )


def check_both_paths(n=48):
    """Both paths on the same inputs, each against the reference with its rounding."""
    case = make_case(n, 8, 256, 2048, 256, torch.bfloat16, hot=0.5)
    hidden, w13, w2, weights, ids = case[:5]
    triton = run(*case, TRITON)
    ref = reference(hidden, w13, w2, weights, ids, round_projections=True)
    torch.testing.assert_close(triton.float(), ref, rtol=RTOL, atol=ATOL)
    if not HAS_NATIVE:
        print("SKIP both paths: the library has no native kernels", flush=True)
        return
    native = run(*case, NATIVE)
    ref_native = reference(hidden, w13, w2, weights, ids, round_projections=False)
    torch.testing.assert_close(native.float(), ref_native, rtol=RTOL, atol=ATOL)
    print(
        f"PASS both paths on shared inputs: n={n}, max|err| native {max_error(native, ref_native):.4f}, "
        f"Triton {max_error(triton, ref):.4f}",
        flush=True,
    )


def main():
    torch.manual_seed(29)
    check("clover-1", 1, 8, 256, 2048, 256, torch.bfloat16)
    check("clover-32", 32, 8, 256, 2048, 256, torch.bfloat16)  # the per-rank Clover shape
    check("clover-128", 128, 8, 256, 2048, 256, torch.bfloat16)
    check("clover-concentrated", 64, 8, 256, 2048, 256, torch.bfloat16, hot=0.8)
    check("clover-fp16", 32, 8, 256, 2048, 256, torch.float16)
    check("clover-dropped-routes", 32, 8, 256, 2048, 256, torch.bfloat16, inactive=3)
    # Rows and k steps that end inside a tile or a 16-byte unit.
    check("ragged-native", 20, 4, 16, 136, 40, torch.bfloat16)
    check("minimal-native", 5, 2, 8, 32, 32, torch.float16)
    # Beyond what the native kernels take: the Triton JIT kernels. (Shapes with exactly two
    # 64-wide k steps, h = 128 say, are left out: the DTK Triton compiler fails on them.)
    check("triton-h", 8, 4, 16, 132, 64, torch.bfloat16)
    check("triton-i", 32, 8, 64, 192, 24, torch.bfloat16)
    check("triton-fp16", 8, 4, 16, 200, 28, torch.float16)
    check_both_paths()


if __name__ == "__main__":
    main()
