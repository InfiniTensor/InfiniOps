import infini.ops as ops
import pytest
import torch


def test_clover_live_rows():
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires Hygon")
    if not hasattr(ops, "CloverLiveMask"):
        pytest.skip("requires Clover providers")
    s = torch.cuda.current_stream().cuda_stream
    ids = torch.tensor([151669, 3, 4, 151669], device="cuda", dtype=torch.int64)
    previous = torch.empty_like(ids)
    live = torch.empty(4, device="cuda", dtype=torch.int8)
    ops.clover_live_mask(ids, previous, 151669, True, live, stream=s)
    assert live.tolist() == [1, 1, 1, 1]
    ids[:] = torch.tensor([5, 3, 4, 151669], device="cuda")
    ops.clover_live_mask(ids, previous, 151669, False, live, stream=s)
    assert live.tolist() == [1, 0, 0, 1]
    routes = torch.arange(16, device="cuda", dtype=torch.int32).view(4, 4)
    ops.clover_mask_routes(routes, live, 256, routes, stream=s)
    assert routes[1:3].eq(256).all()
    fresh = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn_like(fresh)
    old = cache.clone()
    out = torch.empty_like(fresh)
    ops.clover_merge_moe(fresh, live, cache, out, stream=s)
    ref = torch.where(live.bool()[:, None], fresh, old)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    assert torch.equal(cache, ref)
