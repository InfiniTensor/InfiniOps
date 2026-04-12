# Optimization Log — A100-SXM4-80GB

## Round 1: Vectorized Binary Elementwise Brick

**Problem**: Add (4096²) fp16 at 612 GB/s (31% of A100 HBM 2 TB/s).
Each thread processes 1 element, no vectorized load.

**Fix**: Add `BinaryElementwiseVecKernel` with 128-bit coalesced
load/store and grid-stride loop for contiguous tensors.

**Result (DSL Add)**: 612 GB/s → **1646 GB/s** (2.7x, matches PyTorch).

## Round 2: Refactor CudaAdd/CudaSwiglu to Use Vectorized Brick

**Problem**: Hand-written CudaAdd and CudaSwiglu still use old scalar
kernels, not the improved brick.

**Fix**: Replace per-element kernels with `BinaryElementwiseBrick`.

| Operator | Before | After | Speedup |
|----------|--------|-------|---------|
| Add (4096²) fp16 | 0.164 ms (612 GB/s) | 0.077 ms (1315 GB/s) | **2.1x** |
| Swiglu (4096²) fp16 | ~0.164 ms | 0.062 ms (1612 GB/s) | **~2.6x** |

## Round 3: Grid-Stride Loop for Unary Elementwise

**Problem**: Cast fp32→fp16 (4096²) at 626 GB/s.

**Fix**: Add `UnaryElementwiseVecKernel` with grid-stride loop.

**Result**: 0.161 ms (626 GB/s) → **0.092 ms (1094 GB/s)** (1.75x).

## Round 4: RmsNorm Analysis (No Change)

RmsNorm (32,32,4096) is 3.3x slower than PyTorch. Root cause:
PyTorch likely uses a more optimized reduce kernel. Requires deeper
kernel rewrite — deferred.

## Round 5: Post-Optimization Full Benchmark (4096² fp16 on A100)

| Operator | Time (ms) | Bandwidth / TFLOPS | vs PyTorch |
|----------|-----------|-------------------|------------|
| **Add** | 0.076 | 1318 GB/s | 0.80x |
| **Mul** | 0.061 | 1647 GB/s | ≈1.0x |
| **Swiglu** | 0.062 | 1611 GB/s | 1.15x faster |
| **Cast fp32→fp16** | 0.079 | 1279 GB/s | 0.78x |
| **Gemm 4096³** | 0.587 | 234 TFLOPS | ≈1.0x |
| **Matmul 1024³** | 0.017 | 126 TFLOPS | 2.0x faster |
| **Linear 1024×4096²** | 0.171 | — | 1.2x faster |
| **FlashAttn S=2048** | 0.241 | 286 TFLOPS | 1.12x faster |

## Remaining Optimization Opportunities

1. **Add**: 1318 vs PyTorch 1650 GB/s (20% gap) — investigate AddOp
   functor overhead, may need `__hadd2` for fp16 vector operations.
2. **Cast**: 1279 vs 1642 GB/s (22% gap) — needs typed vectorized
   load with different input/output vec sizes.
3. **RmsNorm**: 3.3x slower than PyTorch at (32,32,4096) — needs
   optimized reduce kernel.
4. **Gemm cuBLAS 1024³**: 53 vs PyTorch 62 TFLOPS — switch default to
   cuBLASLt (blocked by test tolerance issue).
