# InfiniOps CUDA Operator Benchmark Baseline

**Date**: 2026-04-11
**Hardware**: NVIDIA A100-SXM4-80GB (SM80)
**CUDA**: 13.0
**Tool**: `torch.utils.benchmark.Timer.blocked_autorange(min_run_time=2)`

---

## Elementwise Operators

| Operator | Shape | dtype | Time (ms) |
|----------|-------|-------|-----------|
| **Add** | (4,4,5632) | fp32 | 0.010 |
| **Add** | (1,32,4096) | fp32 | 0.010 |
| **Add** | (64,32,128) | fp32 | 0.010 |
| **Add** | (4,4,5632) | fp16 | 0.010 |
| **Add** | (1,32,4096) | fp16 | 0.010 |
| **Add** | (64,32,128) | fp16 | 0.010 |
| **Add** | (4,4,5632) | bf16 | 0.010 |
| **Add** | (1,32,4096) | bf16 | 0.010 |
| **Add** | (64,32,128) | bf16 | 0.010 |
| **Mul** | (4,4,5632) | fp32 | 0.010 |
| **Mul** | (1,32,4096) | fp32 | 0.010 |
| **Mul** | (64,32,128) | fp32 | 0.010 |
| **Mul** | (4,4,5632) | fp16 | 0.010 |
| **Mul** | (1,32,4096) | fp16 | 0.010 |
| **Mul** | (64,32,128) | fp16 | 0.010 |
| **Mul** | (4,4,5632) | bf16 | 0.010 |
| **Mul** | (1,32,4096) | bf16 | 0.010 |
| **Mul** | (64,32,128) | bf16 | 0.010 |
| **Cast** | (4,4,5632) | fp32→fp16 | 0.008 |
| **Cast** | (4,4,5632) | fp16→fp32 | 0.008 |
| **Cast** | (1,32,4096) | fp32→bf16 | 0.008 |
| **Cast** | (1,32,4096) | bf16→fp32 | 0.008 |
| **Swiglu** | (4,4,5632) | fp32 | 0.010 |
| **Swiglu** | (1,32,4096) | fp32 | 0.010 |
| **Swiglu** | (4,4,5632) | fp16 | 0.010 |
| **Swiglu** | (1,32,4096) | fp16 | 0.010 |
| **Swiglu** | (4,4,5632) | bf16 | 0.010 |
| **Swiglu** | (1,32,4096) | bf16 | 0.010 |

**Note**: Elementwise ops at these sizes are launch-overhead dominated
(~10 us). Differences become meaningful at larger tensor sizes (>1M
elements).

---

## Normalization Operators

| Operator | Shape | dtype | Time (ms) |
|----------|-------|-------|-----------|
| **RmsNorm** | (2,4,2048) | fp32 | 0.010 |
| **RmsNorm** | (1,32,4096) | fp32 | 0.010 |
| **RmsNorm** | (4,48,64) | fp32 | 0.010 |
| **RmsNorm** | (2,4,2048) | fp16 | 0.010 |
| **RmsNorm** | (1,32,4096) | fp16 | 0.010 |
| **RmsNorm** | (4,48,64) | fp16 | 0.010 |
| **RmsNorm** | (2,4,2048) | bf16 | 0.010 |
| **RmsNorm** | (1,32,4096) | bf16 | 0.010 |
| **RmsNorm** | (4,48,64) | bf16 | 0.010 |
| **AddRmsNorm** | (2,4,2048) | fp32 | 0.014 |
| **AddRmsNorm** | (1,32,4096) | fp32 | 0.014 |
| **AddRmsNorm** | (2,4,2048) | fp16 | 0.014 |
| **AddRmsNorm** | (1,32,4096) | fp16 | 0.014 |
| **AddRmsNorm** | (2,4,2048) | bf16 | 0.014 |
| **AddRmsNorm** | (1,32,4096) | bf16 | 0.014 |
| **CausalSoftmax** | (2,4,64,64) | fp32 | 0.008 |
| **CausalSoftmax** | (1,32,128,128) | fp32 | 0.054 |
| **CausalSoftmax** | (2,4,64,64) | fp16 | 0.008 |
| **CausalSoftmax** | (1,32,128,128) | fp16 | 0.057 |
| **CausalSoftmax** | (2,4,64,64) | bf16 | 0.008 |
| **CausalSoftmax** | (1,32,128,128) | bf16 | 0.061 |

---

## GEMM / Linear

| Operator | Shape (M,N,K) | dtype | Time (ms) | TFLOPS |
|----------|---------------|-------|-----------|--------|
| **Gemm** | (1024,1024,1024) | fp16 | 0.040 | 53.8 |
| **Gemm** | (4096,4096,4096) | fp16 | 0.584 | 235.4 |
| **Gemm** | (1,4096,4096) | fp16 | 0.021 | 1.6 |
| **Gemm** | (1024,1024,1024) | bf16 | 0.038 | 56.0 |
| **Gemm** | (4096,4096,4096) | bf16 | 0.571 | 240.6 |
| **Gemm** | (1,4096,4096) | bf16 | 0.021 | 1.6 |
| **Matmul** | (1024,1024,1024) | fp16 | 0.017 | 124.6 |
| **Matmul** | (4096,4096,4096) | fp16 | 0.590 | 232.9 |
| **Matmul** | (1,4096,4096) | fp16 | 0.023 | 1.5 |
| **Matmul** | (1024,1024,1024) | bf16 | 0.019 | 112.9 |
| **Matmul** | (4096,4096,4096) | bf16 | 0.552 | 248.8 |
| **Matmul** | (1,4096,4096) | bf16 | 0.023 | 1.5 |
| **Linear** | (1024,4096,4096) no bias | fp16 | 0.210 | — |
| **Linear** | (1024,4096,4096) + bias | fp16 | 0.229 | — |
| **Linear** | (1,4096,4096) no bias | fp16 | 0.021 | — |

**Note**: A100 theoretical peak: 312 TFLOPS (fp16 tensor core). Gemm/Matmul
at 4096³ achieve ~235-249 TFLOPS (75-80% utilization). The Matmul 1024³
result (124.6 TFLOPS) is better than Gemm (53.8 TFLOPS) because Matmul
uses cuBLASLt with heuristic algorithm selection.

---

## Position / Cache Operators

| Operator | Config | dtype | Time (ms) |
|----------|--------|-------|-----------|
| **RotaryEmbed** | T=128 H=32 D=128 | fp16 | 0.016 |
| **RotaryEmbed** | T=1 H=32 D=128 | fp16 | 0.016 |
| **RotaryEmbed** | T=512 H=32 D=64 | fp16 | 0.016 |
| **RotaryEmbed** | T=128 H=32 D=128 | bf16 | 0.016 |
| **RotaryEmbed** | T=1 H=32 D=128 | bf16 | 0.016 |
| **RotaryEmbed** | T=512 H=32 D=64 | bf16 | 0.016 |
| **ReshapeAndCache** | T=128 Nkv=8 D=128 BS=16 | fp16 | 0.014 |
| **ReshapeAndCache** | T=32 Nkv=32 D=128 BS=16 | fp16 | 0.014 |

---

## Attention

| Operator | SeqLen | Heads (Q/KV) | HeadDim | dtype | Time (ms) | TFLOPS |
|----------|--------|-------------|---------|-------|-----------|--------|
| **FlashAttn** | 128 | 32/32 | 128 | fp16 | 0.014 | 19.6 |
| **FlashAttn** | 512 | 32/32 | 128 | fp16 | 0.041 | 105.0 |
| **FlashAttn** | 2048 | 32/32 | 128 | fp16 | 0.240 | 286.3 |
| **FlashAttn** | 128 | 32/8 | 128 | fp16 | 0.014 | 19.5 |
| **FlashAttn** | 512 | 32/8 | 128 | fp16 | 0.036 | 119.6 |
| **FlashAttn** | 128 | 32/32 | 128 | bf16 | 0.014 | 19.5 |
| **FlashAttn** | 512 | 32/32 | 128 | bf16 | 0.041 | 105.0 |
| **FlashAttn** | 2048 | 32/32 | 128 | bf16 | 0.240 | 286.6 |
| **FlashAttn** | 128 | 32/8 | 128 | bf16 | 0.014 | 19.7 |
| **FlashAttn** | 512 | 32/8 | 128 | bf16 | 0.036 | 119.7 |

**Note**: FlashAttention via FlashInfer. At S=2048, achieves 286 TFLOPS
(92% of A100 peak). GQA (32/8 heads) is faster than MHA at same seq_len
due to fewer KV heads.

---

## Cat

| Config | dtype | Time (ms) |
|--------|-------|-----------|
| 3×(4,128) dim=0 | fp16 | 0.012 |
| (4,1024)+(4,2048)+(4,512) dim=1 | fp16 | 0.012 |
| 2×(2,32,4096) dim=0 | fp16 | 0.010 |

---

## Optimization Priorities

Based on this baseline, areas with the most optimization potential:

1. **Gemm 1024³**: 53.8 TFLOPS vs Matmul's 124.6 TFLOPS — Gemm uses
   cuBLAS default algorithm while Matmul uses cuBLASLt with heuristic
   search. Consider switching Gemm's default to cuBLASLt.

2. **Linear**: 0.210 ms for (1024,4096,4096) — could benefit from
   cuBLASLt like Matmul.

3. **CausalSoftmax (1,32,128,128)**: 0.054-0.061 ms — relatively slow
   for the size, may benefit from FlashInfer's fused softmax.

4. **Elementwise ops**: All at ~0.010 ms (launch overhead). For larger
   tensors, consider vectorized loads (float4) and grid-stride loops.
