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

## Round 6 (new series): Full Baseline with CUDA Profiler

Used `torch.profiler` to measure actual kernel time (not Python overhead):

| Operator | InfiniOps kernel | PyTorch kernel | Real ratio |
|----------|-----------------|----------------|------------|
| **Add (4096²)** | 60.1 us | 59.3 us | **1.0x ✓** |
| **CausalSoftmax** | 73.3 us | 16.5 us (2 kernels) | **4.4x ✗** |
| **Cast fp32→fp16** | 103.6 us | 61.5 us | **1.7x ✗** |
| **RmsNorm** | 21 us (bench) | 11 us (bench) | **1.9x ✗** |
| **AddRmsNorm** | 42.6 us | 28.9 us (2 kernels) | **1.5x ✗** |

Key insight: Add's 20% benchmark gap is entirely Python binding
overhead — CUDA kernel is matching PyTorch.

## Round 7: Cast Vectorized Load (new series Round 3)

Added 128-bit vectorized input load + output store.

Cast fp32→fp16 (4096²): 0.092 ms → **0.078 ms** (+17%, 1285 GB/s).
Gap vs PyTorch (1645 GB/s): 22% — limited by mixed-type vectorization
(input vec size ≠ output vec size).

## Round 8: RmsNorm Vectorized Attempts (new series Rounds 4-5)

Attempted two approaches:
1. Register caching (store x in registers during reduce, reuse in
   transform) — **failed**: register pressure reduced occupancy, slower.
2. Warp shuffle reduction (replace CUB BlockReduce with manual
   `__shfl_xor_sync`) — **failed**: no improvement, CUB is already
   well-optimized.
3. Vectorized 128-bit struct loads — **failed**: anonymous struct
   alignment issues, compiler couldn't optimize.

Root cause: PyTorch's `vectorized_layer_norm` uses a fundamentally
different approach — needs deeper study with nsight compute.

## Current Status (Post All Optimization)

| Operator | InfiniOps (ms) | PyTorch (ms) | Ratio | Status |
|----------|---------------|-------------|-------|--------|
| Add (4096²) | 0.076 | 0.061 | 0.80x | ✓ kernel matched (binding overhead) |
| Mul (4096²) | 0.061 | 0.061 | 1.00x | ✓ |
| Swiglu (4096²) | 0.062 | 0.167 | 2.68x | ✓ faster |
| Cast (4096²) | 0.078 | 0.061 | 0.78x | ✗ 22% gap |
| RmsNorm | 0.021 | 0.011 | 0.49x | ✗ 2x gap |
| AddRmsNorm | 0.036 | 0.028 | 0.78x | ✗ |
| CausalSoftmax | 0.056 | 0.034 | 0.61x | ✗ |
| Gemm 4096³ | 0.594 | 0.571 | 0.96x | ✓ |
| Matmul 4096³ | 0.590 | 0.574 | 0.97x | ✓ |
| Linear 1024×4096² | 0.173 | 0.211 | 1.22x | ✓ faster |
| RotaryEmbed | 0.020 | 0.099 | 4.93x | ✓ faster |
| FlashAttn S=2048 | 0.240 | 0.269 | 1.12x | ✓ faster |

**7/12 operators match or beat PyTorch.** Remaining gaps in
RmsNorm/AddRmsNorm (vectorized reduce), CausalSoftmax (warp-level
softmax), and Cast (mixed-type vectorization).
