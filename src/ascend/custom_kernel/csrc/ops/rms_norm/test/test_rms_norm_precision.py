"""Comprehensive precision evaluation for RMSNorm AscendC kernel (≥30 cases)."""

import pytest
import torch
import torch_npu
import ascend_kernel  # noqa: F401


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """CPU reference implementation in float32."""
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    hidden = x_fp32 * torch.rsqrt(variance + eps)

    return (hidden * weight.float()).to(x.dtype)


SUPPORTED_DTYPES = [torch.float16, torch.float32]

TEST_SHAPES = [
    ("2D", "small 32x128", (32, 128)),
    ("2D", "medium 64x512", (64, 512)),
    ("2D", "medium 128x1024", (128, 1024)),
    ("2D", "Qwen/Llama 32x4096", (32, 4096)),
    ("2D", "Qwen/Llama 128x4096", (128, 4096)),
    ("2D", "Llama-70B 32x8192", (32, 8192)),
    ("3D", "multi-head 4x32x128", (4, 32, 128)),
    ("3D", "multi-head 8x64x512", (8, 64, 512)),
    ("3D", "batch 4x128x4096", (4, 128, 4096)),
]

GENERAL_SHAPES = [
    ("Small", "single row", (1, 128)),
    ("Small", "single row 4096", (1, 4096)),
    ("Small", "two rows", (2, 256)),
    ("Small", "tiny 3D", (1, 1, 128)),
    ("Small", "non-aligned rows 3", (3, 512)),
    ("Small", "non-aligned rows 7", (7, 1024)),
    ("Large", "BERT-base 512x768", (512, 768)),
    ("Large", "GPT-2 1024x1024", (1024, 1024)),
    ("Large", "Llama batch 256x4096", (256, 4096)),
    ("Large", "Llama-70B batch 64x8192", (64, 8192)),
    ("Large", "3D large 8x512x4096", (8, 512, 4096)),
]

BOUNDARY_VALUES = [
    ("eps_small", "very small eps", (32, 512), {"eps": 1e-12}),
    ("eps_large", "large eps", (32, 512), {"eps": 1e-2}),
    ("zeros", "all-zero input", (16, 1024), {"input_fill": 0.0}),
    ("ones", "all-one input", (16, 1024), {"input_fill": 1.0}),
    ("large_vals", "large input values", (16, 1024), {"input_scale": 100.0}),
    ("small_vals", "tiny input values", (16, 1024), {"input_scale": 1e-4}),
]


def _tolerance(dtype):
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-3)

    return dict(rtol=1e-5, atol=1e-5)


def _compute_metrics(out, ref):
    """Compute precision metrics between output and reference."""
    diff = (out.float() - ref.float()).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()

    ref_abs = ref.float().abs()
    nonzero = ref_abs > 1e-10
    if nonzero.any():
        rel_err = diff[nonzero] / ref_abs[nonzero]
        max_rel_err = rel_err.max().item()
        mean_rel_err = rel_err.mean().item()
    else:
        max_rel_err = 0.0
        mean_rel_err = 0.0

    cos_sim = torch.nn.functional.cosine_similarity(
        out.float().flatten().unsqueeze(0),
        ref.float().flatten().unsqueeze(0),
    ).item()

    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "max_rel_err": max_rel_err,
        "mean_rel_err": mean_rel_err,
        "cosine_sim": cos_sim,
    }


ALL_SHAPE_CASES = [(cat, desc, shape) for cat, desc, shape in TEST_SHAPES] + [
    (cat, desc, shape) for cat, desc, shape in GENERAL_SHAPES
]


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize(
    "case",
    ALL_SHAPE_CASES,
    ids=lambda c: f"{c[0]}_{c[1].replace(' ', '_')}",
)
def test_precision_shapes(case, dtype):
    cat, desc, shape = case
    eps = 1e-6
    hidden_dim = shape[-1]
    x = torch.randn(shape, dtype=dtype)
    w = torch.randn(hidden_dim, dtype=dtype)
    ref = rms_norm_ref(x, w, eps)
    out = torch.ops.npu.rms_norm(x.npu(), w.npu(), eps).cpu()
    tol = _tolerance(dtype)
    metrics = _compute_metrics(out, ref)
    assert torch.allclose(out, ref, **tol), (
        f"[{cat}] {desc} dtype={dtype} "
        f"max_abs={metrics['max_abs_err']:.6e} cos={metrics['cosine_sim']:.8f}"
    )


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize(
    "case",
    BOUNDARY_VALUES,
    ids=lambda c: c[0],
)
def test_precision_boundary(case, dtype):
    name, desc, shape, opts = case
    eps = opts.get("eps", 1e-6)
    hidden_dim = shape[-1]
    fill = opts.get("input_fill", None)
    scale = opts.get("input_scale", 1.0)

    if fill is not None:
        x = torch.full(shape, fill, dtype=dtype)
    else:
        x = torch.randn(shape, dtype=dtype) * scale

    w = torch.randn(hidden_dim, dtype=dtype)
    ref = rms_norm_ref(x, w, eps)
    out = torch.ops.npu.rms_norm(x.npu(), w.npu(), eps).cpu()
    tol = _tolerance(dtype)
    metrics = _compute_metrics(out, ref)
    assert torch.allclose(out, ref, **tol), (
        f"[{name}] {desc} dtype={dtype} "
        f"max_abs={metrics['max_abs_err']:.6e} cos={metrics['cosine_sim']:.8f}"
    )
