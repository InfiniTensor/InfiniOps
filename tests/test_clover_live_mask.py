"""Correctness tests for the shared-layer native Clover metadata operators.

Providers: native SIMT (impl 0, this PR) cross-checked against the Triton
reference (impl 10) on identical inputs, plus explicit torch golden models.
The three ops copy and select values, so everything is compared exactly.
CloverCoreset has its own tests in test_clover_coreset.py.
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


def _require(op):
    """The current CUDA stream once the op has both providers."""
    if not _impls_ok(op):
        pytest.skip("requires CUDA with native and triton providers")
    return torch.cuda.current_stream().cuda_stream


@pytest.mark.parametrize("id_dtype", (torch.int32, torch.int64))
def test_live_mask(device, id_dtype):
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    stream = _require("CloverLiveMask")
    torch.manual_seed(3)
    mask_id, n = 151669, 4096
    ids = torch.randint(0, 200, (n,), device=device, dtype=id_dtype)
    ids[::7] = mask_id
    previous = torch.randint(0, 200, (n,), device=device, dtype=id_dtype)
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
    assert torch.equal(outs[NATIVE][0], golden_live)
    assert torch.equal(outs[NATIVE][1], ids)  # snapshot updated
    assert torch.equal(outs[NATIVE][0], outs[TRITON][0])
    assert torch.equal(outs[NATIVE][1], outs[TRITON][1])

    # refresh=True: everything live regardless of history.
    for impl, prev in ((NATIVE, previous.clone()), (TRITON, previous.clone())):
        infini.ops.clover_live_mask(ids, prev, mask_id, True, live,
                                    stream=stream, implementation_index=impl)
        assert torch.count_nonzero(live) == n, f"impl {impl} refresh"
        assert torch.equal(prev, ids), f"impl {impl} refresh snapshot"


@pytest.mark.parametrize("id_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("topk", (8, 5))  # a power of two, and any other width
def test_mask_routes(device, id_dtype, topk):
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    stream = _require("CloverMaskRoutes")
    torch.manual_seed(4)
    rows, experts = 512, 256
    ids = torch.randint(-1, experts, (rows, topk), device=device, dtype=id_dtype)
    live = (torch.rand(rows, device=device) < 0.3).to(torch.int8)
    out_n = torch.empty_like(ids)
    out_t = torch.empty_like(ids)
    infini.ops.clover_mask_routes(ids, live, experts, out_n,
                                  stream=stream, implementation_index=NATIVE)
    infini.ops.clover_mask_routes(ids, live, experts, out_t,
                                  stream=stream, implementation_index=TRITON)

    golden = torch.where(live.unsqueeze(1) != 0, ids,
                         torch.full_like(ids, experts))
    assert torch.equal(out_n, golden)
    assert torch.equal(out_n, out_t)


def check_merge_moe(stream, fresh, cache0, live):
    out = torch.empty_like(fresh)
    cache_n, cache_t = cache0.clone(), cache0.clone()
    infini.ops.clover_merge_moe(fresh, live, cache_n, out,
                                stream=stream, implementation_index=NATIVE)
    out_n = out.clone()
    infini.ops.clover_merge_moe(fresh, live, cache_t, out,
                                stream=stream, implementation_index=TRITON)
    pick = torch.where(live.unsqueeze(1) != 0, fresh, cache0)
    assert torch.equal(out_n, pick)
    assert torch.equal(cache_n, pick)  # live rows replaced, dead ones kept
    assert torch.equal(out_n, out)
    assert torch.equal(cache_n, cache_t)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("width", (320, 12, 13))
def test_merge_moe(device, dtype, width):
    """Row widths of 320, 12 and 13 elements: 16-byte vectors for every
    element type, for the 32-bit one only, and for none."""
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    stream = _require("CloverMergeMoe")
    torch.manual_seed(5)
    rows = 256
    fresh = torch.randn(rows, width, device=device, dtype=dtype)
    cache0 = torch.randn(rows, width, device=device, dtype=dtype)
    live = (torch.rand(rows, device=device) < 0.4).to(torch.int8)
    check_merge_moe(stream, fresh, cache0, live)


def test_merge_moe_unaligned(device):
    """Tensors that do not start on a 16-byte boundary take one element
    per thread."""
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    stream = _require("CloverMergeMoe")
    torch.manual_seed(8)
    rows, width = 64, 320
    elements = rows * width
    storage = torch.randn(elements + 8, device=device, dtype=torch.bfloat16)
    fresh = storage[1:1 + elements].view(rows, width)  # 2 bytes off
    assert fresh.is_contiguous() and fresh.data_ptr() % 16 != 0
    cache0 = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
    live = (torch.rand(rows, device=device) < 0.4).to(torch.int8)
    check_merge_moe(stream, fresh, cache0, live)
