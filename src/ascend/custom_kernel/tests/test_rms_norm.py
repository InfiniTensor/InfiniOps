"""Functional and precision tests for the RMSNorm AscendC kernel."""

import pytest
import torch
import torch_npu  # noqa: F401  Registers NPU device.
import ascend_kernel  # noqa: F401  Loads libascend_kernel.so into torch.ops.npu.


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """CPU reference implementation in float32."""
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    hidden = x_fp32 * torch.rsqrt(variance + eps)

    return (hidden * weight.float()).to(x.dtype)


DTYPES = [torch.float16, torch.float32]

TEST_SHAPES = [
    (32, 128),
    (64, 512),
    (128, 1024),
    (32, 4096),
    (128, 4096),
    (32, 8192),
    (4, 32, 128),
    (8, 64, 512),
    (4, 128, 4096),
]

GENERAL_SHAPES = [
    (1, 128),
    (1, 4096),
    (2, 256),
    (1, 1, 128),
    (3, 512),
    (7, 1024),
    (512, 768),
    (1024, 1024),
    (256, 4096),
    (64, 8192),
    (8, 512, 4096),
]


def _tolerance(dtype):
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-3)

    return dict(rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize(
    "shape", TEST_SHAPES + GENERAL_SHAPES, ids=lambda s: "x".join(map(str, s))
)
def test_rms_norm_shapes(shape, dtype):
    eps = 1e-6
    hidden_dim = shape[-1]
    x = torch.randn(shape, dtype=dtype)
    w = torch.randn(hidden_dim, dtype=dtype)
    ref = rms_norm_ref(x, w, eps)
    out = torch.ops.npu.rms_norm(x.npu(), w.npu(), eps)
    tol = _tolerance(dtype)
    assert torch.allclose(out.cpu(), ref, **tol), (
        f"shape={shape} dtype={dtype} "
        f"max_abs_err={torch.max(torch.abs(out.cpu() - ref)).item():.6e}"
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize(
    "case",
    [
        ("eps_small", (32, 512), {"eps": 1e-12}),
        ("eps_large", (32, 512), {"eps": 1e-2}),
        ("zeros", (16, 1024), {"input_fill": 0.0}),
        ("ones", (16, 1024), {"input_fill": 1.0}),
        ("large_vals", (16, 1024), {"input_scale": 100.0}),
        ("small_vals", (16, 1024), {"input_scale": 1e-4}),
    ],
    ids=lambda c: c[0],
)
def test_rms_norm_boundary(case, dtype):
    name, shape, opts = case
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
    out = torch.ops.npu.rms_norm(x.npu(), w.npu(), eps)
    tol = _tolerance(dtype)
    assert torch.allclose(out.cpu(), ref, **tol), (
        f"case={name} dtype={dtype} "
        f"max_abs_err={torch.max(torch.abs(out.cpu() - ref)).item():.6e}"
    )


if __name__ == "__main__":
    print("Running quick functional test...")
    x = torch.randn(4, 128, dtype=torch.float16)
    w = torch.randn(128, dtype=torch.float16)
    ref = rms_norm_ref(x, w, 1e-6)
    out = torch.ops.npu.rms_norm(x.npu(), w.npu(), 1e-6)
    max_err = torch.max(torch.abs(out.cpu() - ref)).item()
    print(
        f"  fp16 (4,128): max_abs_err = {max_err:.6e} ... {'PASS' if max_err < 1e-3 else 'FAIL'}"
    )

    x = torch.randn(4, 128, dtype=torch.float32)
    w = torch.randn(128, dtype=torch.float32)
    ref = rms_norm_ref(x, w, 1e-6)
    out = torch.ops.npu.rms_norm(x.npu(), w.npu(), 1e-6)
    max_err = torch.max(torch.abs(out.cpu() - ref)).item()
    print(
        f"  fp32 (4,128): max_abs_err = {max_err:.6e} ... {'PASS' if max_err < 1e-5 else 'FAIL'}"
    )

    print("Quick test done.")
