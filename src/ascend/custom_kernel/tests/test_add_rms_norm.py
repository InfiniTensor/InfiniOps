"""Correctness tests for custom AscendC add_rms_norm kernel."""

import torch
import torch_npu
import pytest


def _load_custom_kernel():
    """Load the custom kernel shared library."""
    import ctypes
    import glob
    import os

    lib_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    libs = glob.glob(os.path.join(lib_dir, "libascend_kernel.so"))
    assert libs, f"No libascend_kernel.so found in {lib_dir}"
    ctypes.CDLL(libs[0])


_load_custom_kernel()


def _ref_add_rms_norm(x1, x2, weight, eps):
    """Reference implementation on CPU (float64 for precision)."""
    x1_f64 = x1.double()
    x2_f64 = x2.double()
    w_f64 = weight.double()

    x_out = x1_f64 + x2_f64
    variance = x_out.pow(2).mean(dim=-1, keepdim=True)
    y = x_out * torch.rsqrt(variance + eps) * w_f64

    return y.to(x1.dtype), x_out.to(x1.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 128),
        (4, 256),
        (8, 512),
        (32, 896),  # Qwen 0.5B hidden_dim.
        (16, 2048),  # Qwen 3B hidden_dim.
        (8, 3584),  # Qwen 7B hidden_dim.
        (1, 4096),  # LLaMA hidden_dim.
        (64, 896),  # Larger batch.
    ],
)
def test_add_rms_norm_correctness(dtype, shape):
    """Verify custom kernel output matches CPU reference."""
    eps = 1e-6
    rows, dim = shape

    x1 = torch.randn(rows, dim, dtype=dtype, device="npu")
    x2 = torch.randn(rows, dim, dtype=dtype, device="npu")
    weight = torch.randn(dim, dtype=dtype, device="npu")

    # Run custom kernel.
    result = torch.ops.npu.add_rms_norm(x1, x2, weight, eps)
    y_npu = result[0]
    x_out_npu = result[1]

    # Run CPU reference.
    y_ref, x_out_ref = _ref_add_rms_norm(x1.cpu(), x2.cpu(), weight.cpu(), eps)

    # Check x_out = x1 + x2.
    rtol_xout = 1e-3 if dtype == torch.float16 else 1e-5
    atol_xout = 1e-3 if dtype == torch.float16 else 1e-5
    assert torch.allclose(x_out_npu.cpu(), x_out_ref, rtol=rtol_xout, atol=atol_xout), (
        f"x_out mismatch: max_diff={(x_out_npu.cpu() - x_out_ref).abs().max().item()}"
    )

    # Check `y = rms_norm(x_out) * weight`.
    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    assert torch.allclose(y_npu.cpu(), y_ref, rtol=rtol, atol=atol), (
        f"y mismatch: max_diff={(y_npu.cpu() - y_ref).abs().max().item()}"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_add_rms_norm_3d(dtype):
    """Verify 3D input (batch, nhead, dim) works correctly."""
    eps = 1e-6
    batch, nhead, dim = 4, 8, 128

    x1 = torch.randn(batch, nhead, dim, dtype=dtype, device="npu")
    x2 = torch.randn(batch, nhead, dim, dtype=dtype, device="npu")
    weight = torch.randn(dim, dtype=dtype, device="npu")

    result = torch.ops.npu.add_rms_norm(x1, x2, weight, eps)
    y_npu = result[0]
    x_out_npu = result[1]

    y_ref, x_out_ref = _ref_add_rms_norm(x1.cpu(), x2.cpu(), weight.cpu(), eps)

    rtol = 1e-3 if dtype == torch.float16 else 1e-5
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    assert torch.allclose(x_out_npu.cpu(), x_out_ref, rtol=rtol, atol=atol)
    assert torch.allclose(y_npu.cpu(), y_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
