"""Times the Hygon `moe_fused_dense`: the native kernels against the Triton JIT ones.

Clover per-rank shapes with TP = 2 by default. Build the adapter first:

    python tests/native/build_clover_workspace.py

usage: python tests/native/bench_moe_fused_dense_hygon.py [--hot SHARE] [tokens ...]

Routing is spread over the experts by default. `--hot` sends a share of the routes to 16
experts instead, the way the block coreset of the model concentrates them, which leaves few
expert blocks to stream.
"""

import infini.ops as ops
import torch
import workspace_ops

BLOCK = 16
DEV = "cuda"
EXPERTS, HIDDEN, INTERMEDIATE, TOPK = 256, 2048, 256, 8
# Bytes of expert weights one layer streams per active expert block.
WEIGHT_BYTES = (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE) * 2
IMPLEMENTATIONS = ((0, "native"), (10, "Triton"))


def milliseconds(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def make_routes(tokens, hot):
    """Routing weights and experts per token. With `hot` > 0 that share of the routes goes to
    16 experts, the way the block coreset of the model concentrates them; the rest is spread."""
    scores = -torch.rand(tokens, EXPERTS, device=DEV)
    if hot > 0:
        crowded = torch.rand(tokens, EXPERTS, device=DEV) < hot
        scores[:, :16] += 2.0 * crowded[:, :16]
    weights, ids = scores.topk(TOPK, dim=-1)
    return weights.float().contiguous(), ids.int().contiguous()


def bench(tokens, hot):
    torch.manual_seed(7)
    weights, ids = make_routes(tokens, hot)
    hidden = torch.randn(tokens, HIDDEN, device=DEV, dtype=torch.bfloat16)
    w13 = torch.randn(EXPERTS, 2 * INTERMEDIATE, HIDDEN, device=DEV, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(EXPERTS, HIDDEN, INTERMEDIATE, device=DEV, dtype=torch.bfloat16) * 0.02
    capacity = tokens * TOPK + EXPERTS * (BLOCK - 1)
    sorted_ids = torch.empty(capacity, device=DEV, dtype=torch.int32)
    experts = torch.empty((capacity + BLOCK - 1) // BLOCK, device=DEV, dtype=torch.int32)
    total = torch.empty(1, device=DEV, dtype=torch.int32)
    stream = torch.cuda.current_stream().cuda_stream
    ops.moe_align_block_size(ids, None, EXPERTS, BLOCK, sorted_ids, experts, total, stream=stream)
    blocks = int(total.item()) // BLOCK
    out = torch.empty_like(hidden)
    tensors = [out, hidden, w13, w2, weights, ids, sorted_ids, experts, total]
    scratch = torch.empty(
        workspace_ops.moe_workspace_size(tensors, BLOCK), device=DEV, dtype=torch.uint8
    )

    results = {}
    for index, name in IMPLEMENTATIONS:
        call = lambda index=index: workspace_ops.moe(tensors, BLOCK, scratch, stream, index)
        results[name] = milliseconds(call)
    line = "  ".join(
        f"{name} {results[name]:7.3f} ms ({blocks * WEIGHT_BYTES / results[name] / 1e6:5.0f} GB/s)"
        for name in results
    )
    print(
        f"tokens {tokens:5d} active blocks {blocks:4d} | {line} | "
        f"{results['Triton'] / results['native']:.2f}x",
        flush=True,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens", nargs="*", type=int, default=[32, 64, 128, 256, 1024])
    parser.add_argument(
        "--hot", type=float, default=0.0, help="share of the routes that go to 16 experts"
    )
    args = parser.parse_args()
    for count in args.tokens:
        bench(count, args.hot)


if __name__ == "__main__":
    main()
