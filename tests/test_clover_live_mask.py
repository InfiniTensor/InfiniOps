"""Correctness tests for the shared-layer native Clover metadata operators.

Providers: native SIMT (impl 0, this PR) cross-checked against the Triton
reference (impl 10) on identical inputs, plus explicit torch golden models.
The first three ops share the `clover_live_rows` semantics; CloverCoreset
recomputes the Triton three-stage pipeline in torch.
"""

import infini.ops
import pytest
import torch

if not hasattr(infini.ops, "CloverLiveMask"):
    pytest.skip("`CloverLiveMask` is not available", allow_module_level=True)

NATIVE, TRITON = 0, 10


def _impls_ok(op):
    impls = getattr(infini.ops, op).active_implementation_indices("nvidia")
    return NATIVE in impls and TRITON in impls


def test_live_mask(device):
    if device != "cuda" or not _impls_ok("CloverLiveMask"):
        pytest.skip("requires CUDA with native and triton providers")
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(3)
    mask_id, n = 151669, 4096
    ids = torch.randint(0, 200, (n,), device=device, dtype=torch.int32)
    ids[::7] = mask_id
    previous = torch.randint(0, 200, (n,), device=device, dtype=torch.int32)
    previous[::5] = mask_id
    live = torch.empty(n, device=device, dtype=torch.int8)

    outs = {}
    for impl, prev in ((NATIVE, previous.clone()), (TRITON, previous.clone())):
        infini.ops.clover_live_mask(ids, prev, mask_id, False, live,
                                    stream=stream, implementation_index=impl)
        outs[impl] = (live.clone(), prev.clone())

    ids_now_mask = ids == mask_id
    prev_was_mask = previous == mask_id
    golden_live = (ids_now_mask | prev_was_mask).to(torch.int8)
    torch.testing.assert_close(outs[NATIVE][0], golden_live)
    torch.testing.assert_close(outs[NATIVE][1], ids)  # snapshot updated
    torch.testing.assert_close(outs[NATIVE][0], outs[TRITON][0])
    torch.testing.assert_close(outs[NATIVE][1], outs[TRITON][1])

    # refresh=True: everything live regardless of history.
    for impl, prev in ((NATIVE, previous.clone()), (TRITON, previous.clone())):
        infini.ops.clover_live_mask(ids, prev, mask_id, True, live,
                                    stream=stream, implementation_index=impl)
        assert torch.count_nonzero(live) == n, f"impl {impl} refresh"


def test_mask_routes(device):
    if device != "cuda" or not _impls_ok("CloverMaskRoutes"):
        pytest.skip("requires CUDA with native and triton providers")
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(4)
    rows, topk, experts = 512, 8, 256
    ids = torch.randint(-1, experts, (rows, topk), device=device, dtype=torch.int32)
    live = (torch.rand(rows, device=device) < 0.3).to(torch.int8)
    out_n = torch.empty_like(ids)
    out_t = torch.empty_like(ids)
    infini.ops.clover_mask_routes(ids, live, experts, out_n,
                                  stream=stream, implementation_index=NATIVE)
    infini.ops.clover_mask_routes(ids, live, experts, out_t,
                                  stream=stream, implementation_index=TRITON)

    golden = torch.where(live.unsqueeze(1) != 0, ids,
                         torch.full_like(ids, experts))
    torch.testing.assert_close(out_n, golden)
    torch.testing.assert_close(out_n, out_t)


def test_merge_moe(device):
    if device != "cuda" or not _impls_ok("CloverMergeMoe"):
        pytest.skip("requires CUDA with native and triton providers")
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(5)
    rows, width = 256, 320
    fresh = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
    cache0 = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
    live = (torch.rand(rows, device=device) < 0.4).to(torch.int8)
    out = torch.empty_like(fresh)

    cache_n, cache_t = cache0.clone(), cache0.clone()
    infini.ops.clover_merge_moe(fresh, live, cache_n, out,
                                stream=stream, implementation_index=NATIVE)
    out_n = out.clone()
    infini.ops.clover_merge_moe(fresh, live, cache_t, out,
                                stream=stream, implementation_index=TRITON)

    pick = torch.where(live.unsqueeze(1) != 0, fresh, cache0)
    torch.testing.assert_close(out_n, pick)
    torch.testing.assert_close(cache_n, pick)  # cache also updated
    torch.testing.assert_close(out_n, out)
    torch.testing.assert_close(cache_n, cache_t)


def torch_coreset(logits, width, top_p):
    """Golden model replicating the Triton three-stage pipeline."""
    tokens, experts = logits.shape
    assert experts == 256
    blocks = tokens // width
    out = torch.empty_like(logits)
    for b in range(blocks):
        block = logits[b * width:(b + 1) * width].float()
        prob = torch.softmax(block, dim=1)
        # top-8 union per row
        sel = torch.zeros_like(prob, dtype=torch.bool)
        rem = block.clone()
        for _ in range(8):
            best = rem.argmax(dim=1)
            sel.scatter_(1, best.unsqueeze(1), True)
            rem.scatter_(1, best.unsqueeze(1), float("-inf"))
        mass = (prob * sel).sum(0)
        order = torch.argsort(mass, descending=True, stable=True)
        p = mass[order] / max(mass.sum(), 1e-9)
        keep_cum = (p.cumsum(0) - p) < top_p
        keep = torch.zeros_like(keep_cum)
        keep[order[keep_cum]] = True
        out[b * width:(b + 1) * width] = torch.where(
            keep.unsqueeze(0), logits[b * width:(b + 1) * width],
            torch.full_like(logits[b * width:(b + 1) * width], float("-inf")))
    return out


def test_coreset(device):
    if device != "cuda" or not _impls_ok("CloverCoreset"):
        pytest.skip("requires CUDA with native and triton providers")
    import importlib.util
    import pathlib
    import sys
    adapter_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "build" / "moe-simt-tests" / "clover_native_adapter.so"
    )
    if not adapter_path.exists():
        pytest.skip("clover_native_adapter is not built; see tests/native/")
    spec = importlib.util.spec_from_file_location("clover_native_adapter", adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules["clover_native_adapter"] = adapter
    spec.loader.exec_module(adapter)

    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(6)
    width = 32
    for blocks, top_p in ((4, 0.9), (2, 0.5), (3, 1.0)):
        logits = torch.randn(blocks * width, 256, device=device, dtype=torch.float32)
        out_n = torch.empty_like(logits)
        out_t = torch.empty_like(logits)
        scratch = torch.empty(blocks * 256 * 8 + blocks * 32 * 16 * 4, device=device, dtype=torch.uint8)
        # Workspace-taking providers are fed via the pybind adapter.
        adapter.coreset(logits, width, top_p, out_n, scratch, stream, NATIVE)
        adapter.coreset(logits, width, top_p, out_t, scratch, stream, TRITON)
        golden = torch_coreset(logits.float().cpu(), width, top_p).to(logits.device)
        torch.testing.assert_close(out_n, golden, equal_nan=True)
        torch.testing.assert_close(out_n, out_t, equal_nan=True)
