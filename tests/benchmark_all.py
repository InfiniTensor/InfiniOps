"""Comprehensive performance benchmark for all CUDA operators.

Run with: pytest tests/benchmark_all.py --benchmark -v -s --devices cuda
"""

import pytest
import torch
import torch.utils.benchmark as benchmark

import infini.ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _bench(fn, label, sub_label, min_run_time=2):
    """Benchmark a function and return the measurement."""
    timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
        label=label,
        sub_label=sub_label,
    )

    return timer.blocked_autorange(min_run_time=min_run_time)


# ---- Add ----

@pytest.mark.benchmark
@pytest.mark.parametrize("shape", [(4, 4, 5632), (1, 32, 4096), (64, 32, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bench_add(shape, dtype):
    a = torch.randn(shape, dtype=dtype, device="cuda")
    b = torch.randn(shape, dtype=dtype, device="cuda")
    out = torch.empty(shape, dtype=dtype, device="cuda")

    m = _bench(lambda: infini.ops.add(a, b, out), "Add", f"{shape} {dtype}")
    print(f"  Add {shape} {dtype}: {m.median*1e3:.3f} ms")


# ---- Mul ----

@pytest.mark.benchmark
@pytest.mark.parametrize("shape", [(4, 4, 5632), (1, 32, 4096), (64, 32, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bench_mul(shape, dtype):
    a = torch.randn(shape, dtype=dtype, device="cuda")
    b = torch.randn(shape, dtype=dtype, device="cuda")
    out = torch.empty(shape, dtype=dtype, device="cuda")

    m = _bench(lambda: infini.ops.mul(a, b, out), "Mul", f"{shape} {dtype}")
    print(f"  Mul {shape} {dtype}: {m.median*1e3:.3f} ms")


# ---- Cast ----

@pytest.mark.benchmark
@pytest.mark.parametrize(
    "shape, in_dtype, out_dtype",
    [
        ((4, 4, 5632), torch.float32, torch.float16),
        ((4, 4, 5632), torch.float16, torch.float32),
        ((1, 32, 4096), torch.float32, torch.bfloat16),
        ((1, 32, 4096), torch.bfloat16, torch.float32),
    ],
)
def test_bench_cast(shape, in_dtype, out_dtype):
    inp = torch.randn(shape, dtype=in_dtype, device="cuda")
    out = torch.empty(shape, dtype=out_dtype, device="cuda")

    m = _bench(
        lambda: infini.ops.cast(inp, out), "Cast", f"{shape} {in_dtype}->{out_dtype}"
    )
    print(f"  Cast {shape} {in_dtype}->{out_dtype}: {m.median*1e3:.3f} ms")


# ---- Swiglu ----

@pytest.mark.benchmark
@pytest.mark.parametrize("shape", [(4, 4, 5632), (1, 32, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bench_swiglu(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device="cuda")
    gate = torch.rand(shape, dtype=dtype, device="cuda")
    out = torch.empty(shape, dtype=dtype, device="cuda")

    m = _bench(
        lambda: infini.ops.swiglu(inp, gate, out), "Swiglu", f"{shape} {dtype}"
    )
    print(f"  Swiglu {shape} {dtype}: {m.median*1e3:.3f} ms")


# ---- RmsNorm ----

@pytest.mark.benchmark
@pytest.mark.parametrize("shape", [(2, 4, 2048), (1, 32, 4096), (4, 48, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bench_rms_norm(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn(shape[-1], dtype=dtype, device="cuda")
    out = torch.empty(shape, dtype=dtype, device="cuda")

    m = _bench(
        lambda: infini.ops.rms_norm(inp, weight, 1e-6, out),
        "RmsNorm",
        f"{shape} {dtype}",
    )
    print(f"  RmsNorm {shape} {dtype}: {m.median*1e3:.3f} ms")


# ---- CausalSoftmax ----

@pytest.mark.benchmark
@pytest.mark.parametrize("shape", [(2, 4, 64, 64), (1, 32, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bench_causal_softmax(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    out = torch.empty(shape, dtype=dtype, device="cuda")

    m = _bench(
        lambda: infini.ops.causal_softmax(inp, out),
        "CausalSoftmax",
        f"{shape} {dtype}",
    )
    print(f"  CausalSoftmax {shape} {dtype}: {m.median*1e3:.3f} ms")


# ---- AddRmsNorm ----

@pytest.mark.benchmark
@pytest.mark.parametrize("shape", [(2, 4, 2048), (1, 32, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bench_add_rms_norm(shape, dtype):
    x1 = torch.randn(shape, dtype=dtype, device="cuda")
    x2 = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn(shape[-1], dtype=dtype, device="cuda")
    y_out = torch.empty(shape, dtype=dtype, device="cuda")
    x_out = torch.empty(shape, dtype=dtype, device="cuda")

    m = _bench(
        lambda: infini.ops.add_rms_norm(x1, x2, weight, 1e-6, y_out, x_out),
        "AddRmsNorm",
        f"{shape} {dtype}",
    )
    print(f"  AddRmsNorm {shape} {dtype}: {m.median*1e3:.3f} ms")


# ---- Cat ----

@pytest.mark.benchmark
@pytest.mark.parametrize(
    "shapes, dim",
    [
        ([(4, 128), (4, 128), (4, 128)], 0),
        ([(4, 1024), (4, 2048), (4, 512)], 1),
        ([(2, 32, 4096), (2, 32, 4096)], 0),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16])
def test_bench_cat(shapes, dim, dtype):
    tensors = [torch.randn(s, dtype=dtype, device="cuda") for s in shapes]

    out_shape = list(shapes[0])
    out_shape[dim] = sum(s[dim] for s in shapes)
    out = torch.empty(out_shape, dtype=dtype, device="cuda")

    first = tensors[0]
    rest = tensors[1:]

    m = _bench(
        lambda: infini.ops.cat(first, rest, dim, out),
        "Cat",
        f"{shapes} dim={dim} {dtype}",
    )
    print(f"  Cat {shapes} dim={dim}: {m.median*1e3:.3f} ms")


# ---- Gemm ----

@pytest.mark.benchmark
@pytest.mark.parametrize(
    "M, N, K",
    [(1024, 1024, 1024), (4096, 4096, 4096), (1, 4096, 4096)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bench_gemm(M, N, K, dtype):
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")
    c = torch.empty(M, N, dtype=dtype, device="cuda")

    m = _bench(
        lambda: infini.ops.gemm(a, b, c), "Gemm", f"({M},{N},{K}) {dtype}"
    )

    tflops = 2 * M * N * K / m.median / 1e12
    print(f"  Gemm ({M},{N},{K}) {dtype}: {m.median*1e3:.3f} ms ({tflops:.1f} TFLOPS)")


# ---- Matmul ----

@pytest.mark.benchmark
@pytest.mark.parametrize(
    "M, N, K",
    [(1024, 1024, 1024), (4096, 4096, 4096), (1, 4096, 4096)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bench_matmul(M, N, K, dtype):
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")
    c = torch.empty(M, N, dtype=dtype, device="cuda")

    m = _bench(
        lambda: infini.ops.matmul(a, b, c, False, False),
        "Matmul",
        f"({M},{N},{K}) {dtype}",
    )

    tflops = 2 * M * N * K / m.median / 1e12
    print(
        f"  Matmul ({M},{N},{K}) {dtype}: {m.median*1e3:.3f} ms ({tflops:.1f} TFLOPS)"
    )


# ---- Linear ----

@pytest.mark.benchmark
@pytest.mark.parametrize(
    "M, N, K, has_bias",
    [(1024, 4096, 4096, False), (1024, 4096, 4096, True), (1, 4096, 4096, False)],
)
@pytest.mark.parametrize("dtype", [torch.float16])
def test_bench_linear(M, N, K, has_bias, dtype):
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")
    bias = torch.randn(N, dtype=dtype, device="cuda") if has_bias else None
    out = torch.empty(M, N, dtype=dtype, device="cuda")

    m = _bench(
        lambda: infini.ops.linear(a, b, bias, False, False, out),
        "Linear",
        f"({M},{N},{K}) bias={has_bias} {dtype}",
    )
    print(
        f"  Linear ({M},{N},{K}) bias={has_bias}: {m.median*1e3:.3f} ms"
    )


# ---- RotaryEmbedding ----

@pytest.mark.benchmark
@pytest.mark.parametrize(
    "num_tokens, num_heads, head_size",
    [(128, 32, 128), (1, 32, 128), (512, 32, 64)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bench_rotary_embedding(num_tokens, num_heads, head_size, dtype):
    positions = torch.arange(num_tokens, device="cuda")
    query = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    key = torch.randn(num_tokens, num_heads, head_size, dtype=dtype, device="cuda")
    cos_sin = torch.randn(8192, head_size, dtype=dtype, device="cuda")
    q_out = torch.empty_like(query)
    k_out = torch.empty_like(key)

    m = _bench(
        lambda: infini.ops.rotary_embedding(
            positions, query, key, cos_sin, head_size, head_size, True, q_out, k_out
        ),
        "RotaryEmbed",
        f"T={num_tokens} H={num_heads} D={head_size} {dtype}",
    )
    print(
        f"  RotaryEmbed T={num_tokens} H={num_heads} D={head_size} {dtype}: "
        f"{m.median*1e3:.3f} ms"
    )


# ---- ReshapeAndCache ----

@pytest.mark.benchmark
@pytest.mark.parametrize(
    "num_tokens, num_kv_heads, head_size, block_size, num_blocks",
    [(128, 8, 128, 16, 64), (32, 32, 128, 16, 32)],
)
@pytest.mark.parametrize("dtype", [torch.float16])
def test_bench_reshape_and_cache(
    num_tokens, num_kv_heads, head_size, block_size, num_blocks, dtype
):
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device="cuda")
    value = torch.randn_like(key)
    kv_cache = torch.zeros(
        2, num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device="cuda"
    )
    slot_mapping = torch.randint(
        0, num_blocks * block_size, (num_tokens,), dtype=torch.int64, device="cuda"
    )
    kv_cache_out = kv_cache.clone()

    m = _bench(
        lambda: infini.ops.reshape_and_cache(
            key, value, kv_cache, slot_mapping, kv_cache_out
        ),
        "ReshapeAndCache",
        f"T={num_tokens} Nkv={num_kv_heads} D={head_size} {dtype}",
    )
    print(
        f"  ReshapeAndCache T={num_tokens} Nkv={num_kv_heads}: {m.median*1e3:.3f} ms"
    )


# ---- FlashAttention ----

@pytest.mark.benchmark
@pytest.mark.parametrize(
    "seq_len, num_heads, num_kv_heads, head_size",
    [
        (128, 32, 32, 128),
        (512, 32, 32, 128),
        (2048, 32, 32, 128),
        (128, 32, 8, 128),
        (512, 32, 8, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bench_flash_attention(seq_len, num_heads, num_kv_heads, head_size, dtype):
    q = torch.randn(seq_len, num_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype, device="cuda")
    v = torch.randn(seq_len, num_kv_heads, head_size, dtype=dtype, device="cuda")
    o = torch.empty(seq_len, num_heads, head_size, dtype=dtype, device="cuda")
    scale = 1.0 / head_size**0.5

    m = _bench(
        lambda: infini.ops.flash_attention(
            q, k, v, None, None, None,
            num_heads, num_kv_heads, head_size, scale,
            True, -1, -1, 0, o,
        ),
        "FlashAttn",
        f"S={seq_len} H={num_heads}/{num_kv_heads} D={head_size} {dtype}",
    )

    # FLOPs: 2 * S * S * H * D (for QK^T) + 2 * S * S * H * D (for attn @ V)
    flops = 4 * seq_len * seq_len * num_heads * head_size
    tflops = flops / m.median / 1e12
    print(
        f"  FlashAttn S={seq_len} H={num_heads}/{num_kv_heads} {dtype}: "
        f"{m.median*1e3:.3f} ms ({tflops:.1f} TFLOPS)"
    )
