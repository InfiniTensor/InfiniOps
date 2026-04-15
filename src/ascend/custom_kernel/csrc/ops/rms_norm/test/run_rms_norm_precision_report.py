"""Generate precision report for RMSNorm AscendC kernel."""

import json
import torch
import torch_npu
import ascend_kernel  # noqa: F401


def rms_norm_ref(x, weight, eps):
    x_fp32 = x.float()
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    hidden = x_fp32 * torch.rsqrt(variance + eps)

    return (hidden * weight.float()).to(x.dtype)


def compute_metrics(out, ref):
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


def run_shape_cases():
    results = []
    all_shapes = TEST_SHAPES + GENERAL_SHAPES

    for cat, desc, shape in all_shapes:
        for dtype in SUPPORTED_DTYPES:
            eps = 1e-6
            hidden_dim = shape[-1]
            x = torch.randn(shape, dtype=dtype)
            w = torch.randn(hidden_dim, dtype=dtype)
            ref = rms_norm_ref(x, w, eps)
            out = torch.ops.npu.rms_norm(x.npu(), w.npu(), eps).cpu()
            m = compute_metrics(out, ref)
            dtype_str = str(dtype).split(".")[-1]

            tol = (1e-3, 1e-3) if dtype == torch.float16 else (1e-5, 1e-5)
            passed = torch.allclose(out, ref, rtol=tol[0], atol=tol[1])

            results.append(
                {
                    "category": cat,
                    "description": desc,
                    "shape": str(shape),
                    "dtype": dtype_str,
                    "max_abs_err": m["max_abs_err"],
                    "mean_abs_err": m["mean_abs_err"],
                    "max_rel_err": m["max_rel_err"],
                    "mean_rel_err": m["mean_rel_err"],
                    "cosine_sim": m["cosine_sim"],
                    "passed": passed,
                }
            )
            status = "PASS" if passed else "FAIL"
            print(
                f"  [{status}] {cat:6s} {desc:30s} {dtype_str:7s} "
                f"max_abs={m['max_abs_err']:.3e} cos={m['cosine_sim']:.8f}"
            )

    return results


def run_boundary_cases():
    results = []

    for name, desc, shape, opts in BOUNDARY_VALUES:
        for dtype in SUPPORTED_DTYPES:
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
            m = compute_metrics(out, ref)
            dtype_str = str(dtype).split(".")[-1]

            tol = (1e-3, 1e-3) if dtype == torch.float16 else (1e-5, 1e-5)
            passed = torch.allclose(out, ref, rtol=tol[0], atol=tol[1])

            results.append(
                {
                    "category": "Boundary",
                    "description": f"{name}: {desc}",
                    "shape": str(shape),
                    "dtype": dtype_str,
                    "max_abs_err": m["max_abs_err"],
                    "mean_abs_err": m["mean_abs_err"],
                    "max_rel_err": m["max_rel_err"],
                    "mean_rel_err": m["mean_rel_err"],
                    "cosine_sim": m["cosine_sim"],
                    "passed": passed,
                }
            )
            status = "PASS" if passed else "FAIL"
            print(
                f"  [{status}] Bound  {name:20s} {dtype_str:7s} "
                f"max_abs={m['max_abs_err']:.3e} cos={m['cosine_sim']:.8f}"
            )

    return results


def main():
    print("=" * 70)
    print("RMSNorm Precision Evaluation Report")
    print("=" * 70)

    print("\n--- Shape Tests ---")
    shape_results = run_shape_cases()

    print("\n--- Boundary Tests ---")
    boundary_results = run_boundary_cases()

    all_results = shape_results + boundary_results
    total = len(all_results)
    passed = sum(1 for r in all_results if r["passed"])

    print(f"\n{'=' * 70}")
    print(f"Summary: {passed}/{total} passed")
    print(f"{'=' * 70}")

    # Save JSON.
    output_path = (
        "/workspace/ascend-kernel/csrc/ops/rms_norm/test/rms_norm_precision.json"
    )
    with open(output_path, "w") as f:
        json.dump(
            {"results": all_results, "total": total, "passed": passed}, f, indent=2
        )

    print(f"JSON report saved to: {output_path}")


if __name__ == "__main__":
    main()
