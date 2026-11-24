"""Correctness tests for the native CloverCoreset provider.

The native kernel mirrors the Triton reference's three stages in one launch
and keeps the association order of every fp32 sum, so the two agree on the
allow list. The torch golden model is an independent statement of the
semantics; its sums associate differently, so it is compared away from the
decision boundary, where a rounding may move one expert.
"""

import importlib.util
import pathlib
import sys

import infini.ops
import pytest
import torch

if not hasattr(infini.ops, "CloverCoreset"):
    pytest.skip("`CloverCoreset` is not available", allow_module_level=True)

NATIVE, TRITON = 0, 10
EXPERTS = 256
TOP_K = 8
# A cumulative mass this close to top_p may fall on either side of it
# depending on how the fp32 sums round.
BOUNDARY = 5.0e-5


def torch_coreset(logits, width, top_p):
    """Golden model of the three-stage pipeline in torch.

    Returns the masked logits and, per expert, the distance between the
    normalized mass ahead of the expert and top_p (zero where no rounding
    can change the decision).
    """
    tokens, experts = logits.shape
    assert experts == EXPERTS
    out = torch.empty_like(logits)
    margin = torch.zeros(tokens, experts, device=logits.device)
    for b in range(tokens // width):
        rows = slice(b * width, (b + 1) * width)
        block = logits[rows].float()
        prob = torch.softmax(block, dim=1)
        # The top-8 union of the rows; argmax takes the first of equal logits.
        selected = torch.zeros_like(prob, dtype=torch.bool)
        remaining = block.clone()
        for _ in range(TOP_K):
            best = remaining.argmax(dim=1)
            selected.scatter_(1, best.unsqueeze(1), True)
            remaining.scatter_(1, best.unsqueeze(1), float("-inf"))
        mass = (prob * selected).sum(0)
        order = torch.argsort(mass, descending=True, stable=True)
        ahead = mass[order].cumsum(0) - mass[order]
        if top_p >= 1.0:
            keep = mass[order] > 0
            keep[0] = True
        else:
            ahead = ahead / max(mass.sum(), 1e-9)
            keep = ahead < top_p
            ahead_expert = torch.empty_like(mass)
            ahead_expert[order] = ahead
            margin[rows] = (ahead_expert - top_p).abs().unsqueeze(0)
        keep_expert = torch.zeros_like(mass, dtype=torch.bool)
        keep_expert[order] = keep
        out[rows] = torch.where(
            keep_expert.unsqueeze(0), logits[rows],
            torch.full_like(logits[rows], float("-inf")))
    return out, margin


def assert_coreset_close(actual, expected, margin):
    """Equal everywhere except at the decision boundary."""
    same = (actual == expected) | (margin <= BOUNDARY)
    assert bool(same.all()), f"{int((~same).sum())} experts differ away from top_p"


def run_native(logits, width, top_p, stream):
    # The native provider needs no workspace: the plain binding serves it.
    out = torch.empty_like(logits)
    infini.ops.clover_coreset(logits, width, top_p, out, stream=stream,
                              implementation_index=NATIVE)
    torch.cuda.synchronize()
    return out


def run_triton(adapter, logits, width, top_p, stream):
    # The Triton provider takes a workspace, which the adapter feeds it.
    out = torch.empty_like(logits)
    blocks = logits.size(0) // width
    scratch = torch.empty(blocks * EXPERTS * 8 + blocks * 32 * 16 * 4,
                          device=logits.device, dtype=torch.uint8)
    adapter.coreset(logits, width, top_p, out, scratch, stream, TRITON)
    torch.cuda.synchronize()
    return out


def require_providers():
    if not _impls_ok():
        pytest.skip("requires CUDA with native and triton providers")
    adapter_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "build" / "moe-simt-tests" / "clover_native_adapter.so"
    )
    if not adapter_path.exists():
        pytest.skip("clover_native_adapter is not built; see tests/native/")
    spec = importlib.util.spec_from_file_location("clover_native_adapter",
                                                  adapter_path)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules["clover_native_adapter"] = adapter
    spec.loader.exec_module(adapter)
    return torch.cuda.current_stream().cuda_stream, adapter


def _impls_ok():
    impls = infini.ops.CloverCoreset.active_implementation_indices("nvidia")
    return NATIVE in impls and TRITON in impls


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("width", (1, 8, 13, 32))
def test_coreset(device, dtype, width):
    """Every block size the operator takes and every element type, over a
    range of routing masses, against the golden model and the reference."""
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    stream, adapter = require_providers()
    torch.manual_seed(6)
    blocks = 3
    logits = (torch.randn(blocks * width, EXPERTS, device=device) * 4).to(dtype)
    before = logits.clone()
    for top_p in (0.05, 0.5, 0.9, 0.999, 1.0):
        out_n = run_native(logits, width, top_p, stream)
        out_t = run_triton(adapter, logits, width, top_p, stream)
        golden, margin = torch_coreset(logits, width, top_p)
        assert_coreset_close(out_n, golden, margin)
        assert_coreset_close(out_t, golden, margin)
        assert_coreset_close(out_n, out_t, margin)
    assert torch.equal(logits, before)  # the operator is out of place


def test_coreset_matches_triton(device):
    """Bit level: the same rounding points leave only the divisions, which
    differ by an ulp and move an expert only at the boundary."""
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    stream, adapter = require_providers()
    torch.manual_seed(61)
    for dtype, width in ((torch.bfloat16, 32), (torch.bfloat16, 7),
                         (torch.float16, 32), (torch.float32, 32)):
        logits = (torch.randn(40 * width, EXPERTS, device=device) * 4).to(dtype)
        out_n = run_native(logits, width, 0.9, stream)
        out_t = run_triton(adapter, logits, width, 0.9, stream)
        _, margin = torch_coreset(logits, width, 0.9)
        assert_coreset_close(out_n, out_t, margin)
        bits = torch.int32 if dtype == torch.float32 else torch.int16
        same = (out_n.view(bits) == out_t.view(bits)).float().mean().item()
        assert same >= 0.9999, f"{dtype} width={width}: only {same:.5%} bit-identical"


def competing_row(logits, row, eighth, ninth):
    """Row `row` of `logits` gets seven high logits; experts 10 and 50
    compete, with the logits `eighth` and `ninth`, for its eighth expert."""
    logits[row] = -10.0
    for column, value in ((3, 5.0), (20, 4.5), (60, 4.0), (100, 3.5),
                          (140, 3.0), (180, 2.5), (200, 2.0)):
        logits[row, column] = value
    logits[row, 10] = eighth
    logits[row, 50] = ninth


def check_competing_rows(logits, adapter, stream):
    """Each of the first four rows is its own routing block. Its eighth
    expert is one of the experts 10 and 50, whichever the rows were given
    the larger logit: the kernel, the reference and the golden model agree
    on it, keep it and drop the other."""
    top_p = 0.999  # far enough to keep the eighth expert of a peaked row
    out_n = run_native(logits, 1, top_p, stream)
    out_t = run_triton(adapter, logits, 1, top_p, stream)
    golden, margin = torch_coreset(logits, 1, top_p)
    assert_coreset_close(out_n, golden, margin)
    assert_coreset_close(out_n, out_t, margin)
    kept_n = torch.isfinite(out_n[:4])
    for out in (out_t, golden):
        assert bool(torch.equal(kept_n, torch.isfinite(out[:4])))
    larger = logits[:4, [10, 50]].float().argmax(dim=1)
    expected = torch.zeros_like(kept_n)
    for row in range(4):
        expected[row, (10, 50)[int(larger[row])]] = True
    kept_pair = kept_n[:, [10, 50]]
    assert bool(torch.equal(kept_pair, expected[:, [10, 50]]))
    return kept_pair


def test_coreset_ties(device):
    """Equal logits are common in 16-bit routing scores. The reference
    breaks ties by the lower expert id and ranks -0.0 with +0.0. Two rows
    have +0.0 at the higher expert id competing with -0.0 for their eighth
    expert."""
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    stream, adapter = require_providers()
    torch.manual_seed(66)
    for dtype in (torch.bfloat16, torch.float16):
        # Coarse logits: many equal ones in the other rows too.
        logits = ((torch.randn(6, EXPERTS, device=device) * 2).to(dtype)
                  .float().mul_(2).round().div_(2))
        for row in range(4):
            if row < 2:
                competing_row(logits, row, -0.0, 0.0)  # the tie goes to 10
            else:
                competing_row(logits, row, -0.0, 1.0)  # 50 is strictly larger
        logits = logits.to(dtype)
        zeros = logits[:2, [10, 50]].float()
        assert bool(torch.signbit(zeros[:, 0]).all())  # -0.0 survived the conversion
        assert not bool(torch.signbit(zeros[:, 1]).any())
        kept_pair = check_competing_rows(logits, adapter, stream)
        assert bool(kept_pair[:2, 0].all()), "the tie goes to the lower expert id"

        # The same rows grouped into wider blocks.
        for width in (2, 3):
            out_n = run_native(logits, width, 0.6, stream)
            out_t = run_triton(adapter, logits, width, 0.6, stream)
            golden, margin = torch_coreset(logits, width, 0.6)
            assert_coreset_close(out_n, golden, margin)
            assert_coreset_close(out_n, out_t, margin)


def test_coreset_near_ties(device):
    """fp32 logits that differ only in the mantissa bits the merge tree's
    keys drop: the exact value still decides, whichever expert id it is at."""
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    stream, adapter = require_providers()
    torch.manual_seed(67)
    logits = (torch.randn(6, EXPERTS, device=device) * 2).float().mul_(2).round().div_(2)
    larger = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)).item()
    assert larger > 1.0
    for row in range(4):
        if row < 2:
            competing_row(logits, row, 1.0, larger)  # the larger logit at the higher id
        else:
            competing_row(logits, row, larger, 1.0)
    kept_pair = check_competing_rows(logits, adapter, stream)
    assert bool(kept_pair[:2, 1].all()), "the larger logit wins at expert 50"
    assert bool(kept_pair[2:, 0].all()), "and at expert 10"


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
def test_coreset_unaligned(device, dtype):
    """A logit tensor that does not start on a 16-byte boundary takes the
    kernel's scalar load and store path."""
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    stream, adapter = require_providers()
    torch.manual_seed(69)
    width, blocks = 8, 3
    storage = torch.randn(blocks * width * EXPERTS + 8, device=device).to(dtype)
    logits = storage[1:1 + blocks * width * EXPERTS].view(blocks * width, EXPERTS)
    assert logits.data_ptr() % 16 != 0 and logits.is_contiguous()
    out_n = run_native(logits, width, 0.7, stream)
    out_t = run_triton(adapter, logits, width, 0.7, stream)
    golden, margin = torch_coreset(logits, width, 0.7)
    assert_coreset_close(out_n, golden, margin)
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    assert torch.equal(out_n.view(bits), out_t.view(bits))


def test_coreset_deterministic(device):
    if device != "cuda":
        pytest.skip("requires the CUDA backend")
    stream, _ = require_providers()
    torch.manual_seed(70)
    for width in (1, 13, 32):
        logits = (torch.randn(6 * width, EXPERTS, device=device) * 4).to(torch.bfloat16)
        first = run_native(logits, width, 0.9, stream)
        for _ in range(4):
            assert torch.equal(run_native(logits, width, 0.9, stream), first), width
