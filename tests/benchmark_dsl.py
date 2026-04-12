"""Performance benchmark comparing DSL-generated vs hand-written kernels.

Measures the execution time of DSL-generated and hand-written (default)
implementations for each operator on CUDA, printing a comparison summary.
"""

import pytest
import torch
import torch.utils.benchmark as benchmark

import infini.ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _setup_binary(shape, dtype, device):
    """Create input, other, and output tensors for binary operators."""
    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)
    out = torch.empty(shape, dtype=dtype, device=device)

    return input, other, out


def _setup_rms_norm(shape, dtype, device):
    """Create input, weight, output tensors and epsilon for RmsNorm."""
    input = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(shape[-1], dtype=dtype, device=device)
    out = torch.empty(shape, dtype=dtype, device=device)
    eps = 1e-6

    return input, weight, out, eps


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_benchmark(fn, label, sub_label, num_warmup=10):
    """Run warmup iterations then measure with ``torch.utils.benchmark.Timer``."""

    for _ in range(num_warmup):
        fn()

    timer = benchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
        label=label,
        sub_label=sub_label,
    )

    return timer.blocked_autorange(min_run_time=1)


def _print_comparison(op_name, shape, dtype, default_result, dsl_result):
    """Print a one-line comparison of default vs DSL timings."""
    default_ms = default_result.median * 1e3
    dsl_ms = dsl_result.median * 1e3
    ratio = default_ms / dsl_ms

    print(
        f"{op_name}: default={default_ms:.3f}ms, dsl={dsl_ms:.3f}ms, "
        f"ratio={ratio:.2f}x  (shape={shape}, dtype={dtype})"
    )


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize("shape", [(4, 4, 5632), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_benchmark_add(shape, dtype):
    """Benchmark Add operator: default (hand-written) vs DSL implementation."""
    device = "cuda"
    input, other, out = _setup_binary(shape, dtype, device)

    label = f"Add {shape} {dtype}"

    default_result = _run_benchmark(
        lambda: infini.ops.add(input, other, out, implementation="default"),
        label,
        "default",
    )

    dsl_result = _run_benchmark(
        lambda: infini.ops.add(input, other, out, implementation="dsl"),
        label,
        "dsl",
    )

    _print_comparison("Add", shape, dtype, default_result, dsl_result)


@pytest.mark.benchmark
@pytest.mark.parametrize("shape", [(4, 4, 5632), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_benchmark_rms_norm(shape, dtype):
    """Benchmark RmsNorm operator: default (hand-written) vs DSL implementation."""
    device = "cuda"
    input, weight, out, eps = _setup_rms_norm(shape, dtype, device)

    label = f"RmsNorm {shape} {dtype}"

    default_result = _run_benchmark(
        lambda: infini.ops.rms_norm(input, weight, eps, out, implementation="default"),
        label,
        "default",
    )

    dsl_result = _run_benchmark(
        lambda: infini.ops.rms_norm(input, weight, eps, out, implementation="dsl"),
        label,
        "dsl",
    )

    _print_comparison("RmsNorm", shape, dtype, default_result, dsl_result)


@pytest.mark.benchmark
@pytest.mark.parametrize("shape", [(4, 4, 5632), (1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_benchmark_swiglu(shape, dtype):
    """Benchmark Swiglu operator: default (hand-written) vs DSL implementation."""
    device = "cuda"
    input, gate, out = _setup_binary(shape, dtype, device)

    label = f"Swiglu {shape} {dtype}"

    default_result = _run_benchmark(
        lambda: infini.ops.swiglu(input, gate, out, implementation="default"),
        label,
        "default",
    )

    dsl_result = _run_benchmark(
        lambda: infini.ops.swiglu(input, gate, out, implementation="dsl"),
        label,
        "dsl",
    )

    _print_comparison("Swiglu", shape, dtype, default_result, dsl_result)
