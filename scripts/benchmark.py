#!/usr/bin/env python
"""InfiniOps Performance Benchmark Script.

Measures performance of native InfiniOps operators and PyTorch fallback
operators against PyTorch reference implementations.

Usage:
    python scripts/benchmark.py --list
    python scripts/benchmark.py --ops add --device cpu --mode quick
    python scripts/benchmark.py --category native --mode standard --output results.json
    python scripts/benchmark.py --mode standard --output full_results.json
"""

import argparse
import json
import pathlib
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `tests.utils` is importable.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.utils.benchmark as benchmark

import infini.ops

from tests.utils import (
    clone_strided,
    empty_strided,
    get_available_devices,
    get_stream,
    randn_strided,
    rand_strided,
    randint_strided,
)


def _clone(obj):
    """Recursively clone tensors/args (mirrors conftest.py:_clone)."""
    if isinstance(obj, torch.Tensor):
        return clone_strided(obj)
    if isinstance(obj, tuple):
        return tuple(_clone(a) for a in obj)
    if isinstance(obj, list):
        return [_clone(a) for a in obj]
    if isinstance(obj, dict):
        return {key: _clone(value) for key, value in obj.items()}
    return obj

from benchmark_configs import (
    FALLBACK_OP_CONFIGS,
    NATIVE_OP_SHAPES,
    NTOPS_OPS,
    NTOPS_TORCH_REF,
    NTOPS_UNARY_SHAPES,
    get_ntops_shapes,
    _DEVICE_ASSERTING_OPS,
    _RANDOM_OPS,
    _SCALAR_VALUES,
    _TENSOR_SHAPES,
    _TYPE_DEFAULTS,
    _VENDOR_SKIP_PATTERNS,
    compute_add_rms_norm_flops,
    compute_data_volume_gb,
    compute_elementwise_flops,
    compute_flash_attention_flops,
    compute_gemm_flops,
    compute_rms_norm_flops,
    compute_rotary_embedding_flops,
    compute_softmax_flops,
    compute_swiglu_flops,
    compute_reshape_and_cache_flops,
    estimate_fallback_flops,
    get_scalar_default,
    num_elements,
    shape_to_str,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PYTORCH_SLOT = 8

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_PLATFORM_TO_TORCH_DEVICE = {
    "nvidia": "cuda",
    "metax": "cuda",
    "iluvatar": "cuda",
    "hygon": "cuda",
    "moore": "musa",
    "cambricon": "mlu",
    "ascend": "npu",
}

# Metadata file paths for torch fallback ops
try:
    _INSTALLED_METADATA_PATH = (
        pathlib.Path(infini.ops.__file__).resolve().with_name("torch_ops_metadata.json")
    )
except Exception:
    _INSTALLED_METADATA_PATH = _PROJECT_ROOT / "generated" / "torch_ops_metadata.json"

_SOURCE_METADATA_PATH = _PROJECT_ROOT / "generated" / "torch_ops_metadata.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    category: str               # "native" or "torch"
    operator: str
    device: str
    dtype: str
    shape_description: str
    description: str
    infiniops_median_us: float
    infiniops_mean_us: float
    infiniops_std_us: float
    reference_median_us: float
    reference_mean_us: float
    reference_std_us: float
    speedup: float
    tflops: float = 0.0
    throughput_gb_s: float = 0.0
    num_iterations: int = 0
    status: str = "ok"          # "ok", "skip", "error"
    message: str = ""


@dataclass
class BenchmarkConfig:
    mode: str = "standard"
    warmup: int = 3
    min_time: float = 0.1
    devices: list = field(default_factory=list)
    dtypes: list = field(default_factory=list)
    ops: list = field(default_factory=list)
    category: str = "all"
    device_display_names: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Measurement engine
# ---------------------------------------------------------------------------

def _synchronize(device):
    """Synchronize the device stream."""
    if device == "cpu":
        return
    mod = getattr(torch, device, None)
    if mod is not None and hasattr(mod, "synchronize"):
        mod.synchronize()


def measure(infiniops_fn, ref_fn, args, kwargs, ref_args, ref_kwargs,
            device, warmup=3, min_time=0.1):
    """Measure InfiniOps vs reference performance.

    Returns (infiniops_stats, ref_stats) where each is (median_us, mean_us, std_us, n_iters).
    """
    # Warmup
    for _ in range(warmup):
        infiniops_fn(*args, **kwargs)
    _synchronize(device)

    if device == "cpu":
        # CPU: use blocked_autorange (ops are synchronous)
        stmt = "func(*args, **kwargs)"
        func_timer = benchmark.Timer(
            stmt=stmt,
            globals={"func": infiniops_fn, "args": args, "kwargs": kwargs},
        )
        func_measurement = func_timer.blocked_autorange(min_run_time=min_time)

        ref_timer = benchmark.Timer(
            stmt=stmt,
            globals={"func": ref_fn, "args": ref_args, "kwargs": ref_kwargs},
        )
        ref_measurement = ref_timer.blocked_autorange(min_run_time=min_time)

        func_stats = (
            func_measurement.median * 1e6,
            func_measurement.mean * 1e6,
            func_measurement.iqr * 1e6,
            len(func_measurement.times),
        )
        ref_stats = (
            ref_measurement.median * 1e6,
            ref_measurement.mean * 1e6,
            ref_measurement.iqr * 1e6,
            len(ref_measurement.times),
        )
    else:
        # GPU/MLU/NPU: per-iteration sync to measure actual execution time.
        # Each iteration: sync → start timer → execute → sync → stop timer.
        n_iters = 100

        # Warmup for InfiniOps
        for _ in range(warmup):
            infiniops_fn(*args, **kwargs)
        _synchronize(device)

        total = 0.0
        for _ in range(n_iters):
            _synchronize(device)
            t0 = time.perf_counter()
            infiniops_fn(*args, **kwargs)
            _synchronize(device)
            total += time.perf_counter() - t0
        infiniops_us = total / n_iters * 1e6

        # Warmup for reference
        for _ in range(warmup):
            ref_fn(*ref_args, **ref_kwargs)
        _synchronize(device)

        total = 0.0
        for _ in range(n_iters):
            _synchronize(device)
            t0 = time.perf_counter()
            ref_fn(*ref_args, **ref_kwargs)
            _synchronize(device)
            total += time.perf_counter() - t0
        ref_us = total / n_iters * 1e6

        func_stats = (infiniops_us, infiniops_us, 0.0, n_iters)
        ref_stats = (ref_us, ref_us, 0.0, n_iters)

    return func_stats, ref_stats


# ---------------------------------------------------------------------------
# Operator discovery
# ---------------------------------------------------------------------------

def discover_native_operators():
    """Return list of available native operator names."""
    available = []
    for name in NATIVE_OP_SHAPES:
        op_pascal = "".join(part.capitalize() for part in name.split("_"))
        op_cls = getattr(infini.ops, op_pascal, None)
        if op_cls is not None and hasattr(op_cls, "active_implementation_indices"):
            available.append(name)
    return available


def discover_fallback_operators():
    """Return list of available fallback operator names from metadata."""
    metadata = _load_metadata()
    if not metadata:
        return []
    ops = []
    seen = set()
    for op_meta in metadata.get("ops", []):
        aten_name = op_meta.get("aten_name", op_meta["name"])
        if aten_name in seen:
            continue
        seen.add(aten_name)
        ops.append(op_meta["name"])
    return ops


def _load_metadata():
    """Load torch_ops_metadata.json if available."""
    for path in (_INSTALLED_METADATA_PATH, _SOURCE_METADATA_PATH):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
    return None


def _get_op_class(op_name):
    """Get the InfiniOps operator class from snake_case name."""
    op_pascal = "".join(part.capitalize() for part in op_name.split("_"))
    return getattr(infini.ops, op_pascal, None)


# ---------------------------------------------------------------------------
# Native operator setup functions
# ---------------------------------------------------------------------------

def setup_add(device, dtype, shape_args):
    shape = shape_args
    input_t = randn_strided(shape, None, dtype=dtype, device=device)
    other_t = randn_strided(shape, None, dtype=dtype, device=device)
    out = empty_strided(shape, None, dtype=dtype, device=device)

    def infiniops_fn(input, other, out):
        infini.ops.add(input, other, out, stream=get_stream(input.device))
        return out

    def ref_fn(input, other, out):
        torch.add(input, other, out=out)
        return out

    return infiniops_fn, ref_fn, (input_t, other_t, out), {}


def setup_mul(device, dtype, shape_args):
    shape = shape_args
    input_t = randn_strided(shape, None, dtype=dtype, device=device)
    other_t = randn_strided(shape, None, dtype=dtype, device=device)
    out = empty_strided(shape, None, dtype=dtype, device=device)

    def infiniops_fn(input, other, out):
        infini.ops.mul(input, other, out, stream=get_stream(input.device))
        return out

    def ref_fn(input, other, out):
        torch.mul(input, other, out=out)
        return out

    return infiniops_fn, ref_fn, (input_t, other_t, out), {}


def setup_cast(device, dtype, shape_args):
    shape = shape_args
    input_t = randn_strided(shape, None, dtype=dtype, device=device)
    # Cast to the opposite type for benchmarking
    out_dtype = torch.float32 if dtype == torch.float16 else torch.float16
    out = empty_strided(shape, None, dtype=out_dtype, device=device)

    def infiniops_fn(input, out):
        infini.ops.cast(input, out, stream=get_stream(input.device))
        return out

    def ref_fn(input, out):
        out.copy_(input.to(out.dtype))
        return out

    return infiniops_fn, ref_fn, (input_t, out), {}


def setup_cat(device, dtype, shape_args):
    # shape_args is (tuple_of_shapes, dim)
    shapes, dim = shape_args[:-1], shape_args[-1]
    inputs = [randn_strided(s, None, dtype=dtype, device=device) for s in shapes]
    out_shape = list(shapes[0])
    out_shape[dim % len(out_shape)] = sum(s[dim % len(s)] for s in shapes)
    out = empty_strided(out_shape, None, dtype=dtype, device=device)

    all_args = (*inputs, out)

    def infiniops_fn(*args, dim=dim):
        inps = list(args[:-1])
        o = args[-1]
        first = inps[0]
        rest = inps[1:]
        infini.ops.cat(first, rest, dim, o, stream=get_stream(first.device))
        return o

    def ref_fn(*args, dim=dim):
        inps = list(args[:-1])
        o = args[-1]
        result = torch.cat(inps, dim=dim)
        o.copy_(result)
        return o

    return infiniops_fn, ref_fn, all_args, {}


def setup_gemm(device, dtype, shape_args):
    a_shape, b_shape = shape_args
    M, K = a_shape
    _, N = b_shape
    c_shape = a_shape[:-1] + b_shape[-1:]
    a = randn_strided(a_shape, None, dtype=dtype, device=device)
    b = randn_strided(b_shape, None, dtype=dtype, device=device)
    c = randn_strided(c_shape, None, dtype=dtype, device=device)
    alpha, beta = 1.0, 0.0

    def infiniops_fn(a, b, alpha, beta, trans_a, trans_b, c):
        infini.ops.gemm(a, b, alpha, beta, trans_a, trans_b, c,
                        stream=get_stream(a.device))
        return c

    def ref_fn(a, b, alpha, beta, trans_a, trans_b, c):
        if alpha == 0:
            c.mul_(beta)
            return c
        result = torch.matmul(a.float(), b.float())
        c.copy_((alpha * result + beta * c.float()).to(c.dtype))
        return c

    return infiniops_fn, ref_fn, (a, b, alpha, beta, False, False, c), {}


def setup_matmul(device, dtype, shape_args):
    a_shape, b_shape = shape_args
    c_shape = a_shape[:-1] + b_shape[-1:]
    a = randn_strided(a_shape, None, dtype=dtype, device=device)
    b = randn_strided(b_shape, None, dtype=dtype, device=device)
    c = empty_strided(c_shape, None, dtype=dtype, device=device)

    def infiniops_fn(a, b, c, trans_a=False, trans_b=False):
        infini.ops.matmul(a, b, c, trans_a, trans_b, stream=get_stream(a.device))
        return c

    def ref_fn(a, b, c, trans_a=False, trans_b=False):
        result = torch.matmul(a.float(), b.float()).to(c.dtype)
        c.copy_(result)
        return c

    return infiniops_fn, ref_fn, (a, b, c), {}


def setup_linear(device, dtype, shape_args):
    a_shape, b_shape = shape_args
    out_shape = a_shape[:-1] + b_shape[-1:]
    a = randn_strided(a_shape, None, dtype=dtype, device=device)
    b = randn_strided(b_shape, None, dtype=dtype, device=device)
    bias = randn_strided((b_shape[-1],), None, dtype=dtype, device=device)
    out = empty_strided(out_shape, None, dtype=dtype, device=device)

    def infiniops_fn(a, b, bias, out, trans_a=False, trans_b=False):
        infini.ops.linear(a, b, bias, trans_a, trans_b, out,
                          stream=get_stream(a.device))
        return out

    def ref_fn(a, b, bias, out, trans_a=False, trans_b=False):
        result = torch.matmul(a.float(), b.float())
        if bias is not None:
            result = result + bias.float()
        out.copy_(result.to(out.dtype))
        return out

    return infiniops_fn, ref_fn, (a, b, bias, out), {}


def setup_rms_norm(device, dtype, shape_args):
    input_shape = shape_args
    weight_shape = (input_shape[-1],)
    input_t = randn_strided(input_shape, None, dtype=dtype, device=device)
    weight = randn_strided(weight_shape, None, dtype=dtype, device=device)
    out = empty_strided(input_shape, None, dtype=dtype, device=device)
    eps = 1e-6

    def infiniops_fn(input, weight, out, eps=1e-6):
        infini.ops.rms_norm(input, weight, eps, out,
                            stream=get_stream(input.device))
        return out

    def ref_fn(input, weight, out, eps=1e-6):
        rms_norm_fn = getattr(torch.nn.functional, "rms_norm", None)
        if rms_norm_fn is None:
            rms = torch.sqrt(torch.mean(input * input, dim=-1, keepdim=True) + eps)
            result = (input / rms) * weight
        else:
            result = rms_norm_fn(input, input.shape[-1:], weight=weight, eps=eps)
        out.copy_(result)
        return out

    return infiniops_fn, ref_fn, (input_t, weight, out), {"eps": eps}


def setup_causal_softmax(device, dtype, shape_args):
    shape = shape_args
    input_t = randn_strided(shape, None, dtype=dtype, device=device)
    out = empty_strided(shape, None, dtype=dtype, device=device)

    def infiniops_fn(input, out):
        infini.ops.causal_softmax(input, out, stream=get_stream(input.device))
        return out

    def ref_fn(input, out):
        input_cpu = input.detach().cpu().to(torch.float32)
        mask = torch.tril(torch.ones_like(input_cpu), diagonal=-1).flip(dims=[-2, -1])
        masked = torch.where(mask == 1, -torch.inf, input_cpu)
        result = torch.nn.functional.softmax(masked, dim=-1)
        out.copy_(result.to(device=out.device, dtype=out.dtype))
        return out

    return infiniops_fn, ref_fn, (input_t, out), {}


def setup_swiglu(device, dtype, shape_args):
    shape = shape_args
    input_t = rand_strided(shape, None, dtype=dtype, device=device)
    gate = rand_strided(shape, None, dtype=dtype, device=device)
    out = empty_strided(shape, None, dtype=dtype, device=device)

    def infiniops_fn(input, gate, out):
        infini.ops.swiglu(input, gate, out, stream=get_stream(input.device))
        return out

    def ref_fn(input, gate, out):
        swish_x = gate * torch.sigmoid(gate)
        torch.mul(input, swish_x, out=out)
        return out

    return infiniops_fn, ref_fn, (input_t, gate, out), {}


def setup_flash_attention(device, dtype, shape_args):
    q_shape, kv_shape, seq_len = shape_args[0], shape_args[1], shape_args[2]
    num_heads = q_shape[1]
    num_kv_heads = kv_shape[1]
    head_size = q_shape[2]
    scale = 1.0 / (head_size ** 0.5)

    query = randn_strided(q_shape, None, dtype=dtype, device=device)
    key = randn_strided(kv_shape, None, dtype=dtype, device=device)
    value = randn_strided(kv_shape, None, dtype=dtype, device=device)
    output = empty_strided(q_shape, None, dtype=dtype, device=device)

    def infiniops_fn(query, key, value, output, **kw):
        infini.ops.flash_attention(
            query, key, value,
            None,   # cu_seqlens_q (not used for decode)
            None,   # cu_seqlens_kv
            None,   # block_table
            num_heads, num_kv_heads, head_size, scale,
            True,   # causal
            -1,     # window_left
            -1,     # window_right
            0,      # block_size
            output,
            stream=get_stream(query.device),
        )
        return output

    def ref_fn(query, key, value, output, **kw):
        # Simple scaled dot product attention reference
        # query: [T, H, D], key: [T, KV_H, D]
        T = query.shape[0]
        H = num_heads
        KV_H = num_kv_heads
        D = head_size
        S = T  # self-attention
        G = H // KV_H  # group size

        q = query.float().view(T, KV_H, G, D)
        k = key.float().view(T, KV_H, D)
        v = value.float().view(T, KV_H, D)

        # attn_weights: [KV_H, G, T, S]
        attn_weights = torch.einsum("tkgd,shd->tgts", q, k) * scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, S, device=query.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], -torch.inf)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # attn_output: [KV_H, G, T, D]
        attn_output = torch.einsum("tgts,shd->tkgd", attn_weights, v)
        result = attn_output.reshape(T, H, D).to(dtype)
        output.copy_(result)
        return output

    return infiniops_fn, ref_fn, (query, key, value, output), {}


def setup_rotary_embedding(device, dtype, shape_args):
    q_shape = shape_args[0]
    kv_shape = shape_args[1]
    seq_len = shape_args[2]
    head_size = shape_args[3]
    rotary_dim = head_size
    num_heads = q_shape[1]
    num_kv_heads = kv_shape[1]

    positions = randint_strided(0, seq_len, (q_shape[0],), None,
                                dtype=torch.int64, device=device)
    query = randn_strided(q_shape, None, dtype=dtype, device=device)
    key = randn_strided(kv_shape, None, dtype=dtype, device=device)

    # cos_sin_cache: [max_seq_len, rotary_dim]
    cos_sin_cache = randn_strided(
        (seq_len, rotary_dim), None, dtype=dtype, device=device
    )
    query_out = empty_strided(q_shape, None, dtype=dtype, device=device)
    key_out = empty_strided(kv_shape, None, dtype=dtype, device=device)

    def infiniops_fn(positions, query, key, cos_sin_cache, query_out, key_out):
        infini.ops.rotary_embedding(
            positions, query, key, cos_sin_cache,
            head_size, rotary_dim, True,  # is_neox_style
            query_out, key_out,
            stream=get_stream(query.device),
        )
        return query_out, key_out

    def ref_fn(positions, query, key, cos_sin_cache, query_out, key_out):
        # Simplified reference — just copy for timing purposes
        query_out.copy_(query)
        key_out.copy_(key)
        return query_out, key_out

    return infiniops_fn, ref_fn, (positions, query, key, cos_sin_cache,
                                  query_out, key_out), {}


def setup_add_rms_norm(device, dtype, shape_args):
    shape = shape_args
    input_t = randn_strided(shape, None, dtype=dtype, device=device)
    other = randn_strided(shape, None, dtype=dtype, device=device)
    weight = randn_strided((shape[-1],), None, dtype=dtype, device=device)
    eps = 1e-6

    if len(shape) == 2:
        rstd_shape = (shape[0],)
    else:
        rstd_shape = (shape[0], shape[1])

    out = empty_strided(shape, None, dtype=dtype, device=device)
    rstd_out = empty_strided(rstd_shape, None, dtype=dtype, device=device)

    def infiniops_fn(input, other, weight, out, rstd_out, eps=1e-6):
        infini.ops.add_rms_norm(input, other, weight, eps, out, rstd_out,
                                stream=get_stream(input.device))
        return out

    def ref_fn(input, other, weight, out, rstd_out, eps=1e-6):
        added = input + other
        rms_norm_fn = getattr(torch.nn.functional, "rms_norm", None)
        if rms_norm_fn is None:
            rms = torch.sqrt(torch.mean(added * added, dim=-1, keepdim=True) + eps)
            result = (added / rms) * weight
        else:
            result = rms_norm_fn(added, added.shape[-1:], weight=weight, eps=eps)
        out.copy_(result)
        return out

    return infiniops_fn, ref_fn, (input_t, other, weight, out, rstd_out), {"eps": eps}


def setup_reshape_and_cache(device, dtype, shape_args):
    kv_shape = shape_args
    num_tokens, num_kv_heads, head_size = kv_shape

    key = randn_strided(kv_shape, None, dtype=dtype, device=device)
    value = randn_strided(kv_shape, None, dtype=dtype, device=device)

    block_size = 16
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    kv_cache_shape = (2, num_blocks, block_size, num_kv_heads, head_size)
    kv_cache = torch.zeros(kv_cache_shape, dtype=dtype, device=device)
    kv_cache_out = torch.zeros_like(kv_cache)

    # slot_mapping: random slots within [0, num_blocks * block_size)
    max_slots = num_blocks * block_size
    slot_mapping = randint_strided(0, max_slots, (num_tokens,), None,
                                   dtype=torch.int64, device=device)

    def infiniops_fn(key, value, kv_cache, slot_mapping, kv_cache_out):
        kv_cache_out.copy_(kv_cache)
        infini.ops.reshape_and_cache(key, value, kv_cache_out, slot_mapping,
                                     kv_cache_out,
                                     stream=get_stream(key.device))
        return kv_cache_out

    def ref_fn(key, value, kv_cache, slot_mapping, kv_cache_out):
        # Simple reference: scatter key/value into cache
        kv_cache_out.copy_(kv_cache)
        for t in range(num_tokens):
            slot = slot_mapping[t].item()
            block_idx = slot // block_size
            offset = slot % block_size
            kv_cache_out[0, block_idx, offset, :, :] = key[t, :, :]
            kv_cache_out[1, block_idx, offset, :, :] = value[t, :, :]
        return kv_cache_out

    return infiniops_fn, ref_fn, (key, value, kv_cache, slot_mapping, kv_cache_out), {}


# Setup function registry
_NATIVE_SETUP = {
    "add": setup_add,
    "mul": setup_mul,
    "cast": setup_cast,
    "cat": setup_cat,
    "gemm": setup_gemm,
    "matmul": setup_matmul,
    "linear": setup_linear,
    "rms_norm": setup_rms_norm,
    "causal_softmax": setup_causal_softmax,
    "swiglu": setup_swiglu,
    "flash_attention": setup_flash_attention,
    "rotary_embedding": setup_rotary_embedding,
    "add_rms_norm": setup_add_rms_norm,
    "reshape_and_cache": setup_reshape_and_cache,
}


# ---------------------------------------------------------------------------
# Native operator benchmark runner
# ---------------------------------------------------------------------------

def run_native_benchmarks(config):
    """Run benchmarks for native operators."""
    results = []
    available_ops = discover_native_operators()

    # Filter ops
    if config.ops:
        available_ops = [op for op in available_ops if op in config.ops]

    total = len(available_ops)
    for idx, op_name in enumerate(available_ops, 1):
        setup_fn = _NATIVE_SETUP.get(op_name)
        if setup_fn is None:
            continue

        op_shapes = NATIVE_OP_SHAPES.get(op_name, {})
        shapes_list = op_shapes.get(config.mode, op_shapes.get("quick", []))

        for device in config.devices:
            for dtype in config.dtypes:
                for shape_entry in shapes_list:
                    if isinstance(shape_entry[-1], str):
                        shape_args = shape_entry[:-1]
                        description = shape_entry[-1]
                    else:
                        shape_args = shape_entry
                        description = ""

                    # Unwrap single-element tuple: ((1, 4096),) -> (1, 4096)
                    if (isinstance(shape_args, tuple) and len(shape_args) == 1
                            and isinstance(shape_args[0], (tuple, list))):
                        shape_args = shape_args[0]

                    # Shape description for output
                    if len(shape_args) == 1:
                        shape_desc = shape_to_str(shape_args[0])
                    elif len(shape_args) == 2:
                        shape_desc = (shape_to_str(shape_args[0]) + "x"
                                      + shape_to_str(shape_args[1]))
                    else:
                        shape_desc = shape_to_str(shape_args)

                    try:
                        result = _run_single_native(
                            op_name, setup_fn, device, dtype,
                            shape_args, shape_desc, description, config,
                        )
                        results.append(result)
                    except Exception as exc:
                        results.append(BenchmarkResult(
                            category="native",
                            operator=op_name,
                            device=device,
                            dtype=str(dtype),
                            shape_description=shape_desc,
                            description=description,
                            infiniops_median_us=0, infiniops_mean_us=0,
                            infiniops_std_us=0,
                            reference_median_us=0, reference_mean_us=0,
                            reference_std_us=0,
                            speedup=0,
                            status="error",
                            message=str(exc),
                        ))

        # Progress log
        op_ok = sum(1 for r in results if r.operator == op_name and r.status == "ok")
        op_err = sum(1 for r in results if r.operator == op_name and r.status != "ok")
        print(f"  [{idx}/{total}] {op_name:<15}   {op_ok} ok, {op_err} err")

    return results


def _run_single_native(op_name, setup_fn, device, dtype, shape_args,
                        shape_desc, description, config):
    """Run a single native op benchmark."""
    # Check if op is available on device
    op_cls = _get_op_class(op_name)
    if op_cls is None or not op_cls.active_implementation_indices(device):
        return BenchmarkResult(
            category="native", operator=op_name, device=device,
            dtype=str(dtype), shape_description=shape_desc,
            description=description,
            infiniops_median_us=0, infiniops_mean_us=0, infiniops_std_us=0,
            reference_median_us=0, reference_mean_us=0, reference_std_us=0,
            speedup=0, status="skip",
            message=f"No implementation on {device}",
        )

    try:
        infiniops_fn, ref_fn, args, kwargs = setup_fn(device, dtype, shape_args)
    except RuntimeError as exc:
        return BenchmarkResult(
            category="native", operator=op_name, device=device,
            dtype=str(dtype), shape_description=shape_desc,
            description=description,
            infiniops_median_us=0, infiniops_mean_us=0, infiniops_std_us=0,
            reference_median_us=0, reference_mean_us=0, reference_std_us=0,
            speedup=0, status="skip",
            message=f"Setup failed: {exc}",
        )

    # Clone args for reference
    ref_args = _clone(args)
    ref_kwargs = _clone(kwargs) if kwargs else {}

    try:
        func_stats, ref_stats = measure(
            infiniops_fn, ref_fn, args, kwargs, ref_args, ref_kwargs,
            device, warmup=config.warmup, min_time=config.min_time,
        )
    except Exception as exc:
        return BenchmarkResult(
            category="native", operator=op_name, device=device,
            dtype=str(dtype), shape_description=shape_desc,
            description=description,
            infiniops_median_us=0, infiniops_mean_us=0, infiniops_std_us=0,
            reference_median_us=0, reference_mean_us=0, reference_std_us=0,
            speedup=0, status="error",
            message=f"Measurement failed: {exc}",
        )

    ininiops_median, infiniops_mean, infiniops_std, n_iters = func_stats
    ref_median, ref_mean, ref_std, _ = ref_stats

    speedup = ref_median / ininiops_median if ininiops_median > 0 else 0

    # Clear operator cache
    if op_cls is not None and hasattr(op_cls, "clear_cache"):
        op_cls.clear_cache()

    return BenchmarkResult(
        category="native",
        operator=op_name,
        device=device,
        dtype=str(dtype),
        shape_description=shape_desc,
        description=description,
        infiniops_median_us=round(ininiops_median, 2),
        infiniops_mean_us=round(infiniops_mean, 2),
        infiniops_std_us=round(infiniops_std, 2),
        reference_median_us=round(ref_median, 2),
        reference_mean_us=round(ref_mean, 2),
        reference_std_us=round(ref_std, 2),
        speedup=round(speedup, 2),
        num_iterations=n_iters,
    )


# ---------------------------------------------------------------------------
# Fallback operator benchmark runner
# ---------------------------------------------------------------------------

def _torch_func(op_name):
    """Resolve the reference function for a fallback op."""
    if op_name.endswith("_") and not op_name.endswith("__"):
        # Inplace
        method_name = op_name
        def _call_inplace(input, *args):
            return getattr(input, method_name)(*args)
        return _call_inplace

    for namespace in (torch, torch.special, torch.nn.functional):
        func = getattr(namespace, op_name, None)
        if func is not None:
            return func

    if op_name.startswith("special_"):
        func = getattr(torch.special, op_name.removeprefix("special_"), None)
        if func is not None:
            return func

    return None


def _build_fallback_inputs(op_name, op_meta, shapes, dtype, device):
    """Build inputs for a fallback op based on metadata."""
    in_params = [p for p in op_meta["params"] if not p.get("is_out", False)]

    inputs = []
    tensor_idx = 0
    for p in in_params:
        if p.get("is_tensor", False):
            per_op = _TENSOR_SHAPES.get(op_name)
            if per_op is not None and tensor_idx < len(per_op):
                tshape = per_op[tensor_idx]
            else:
                # Use provided shapes (may be a single shape or per-tensor)
                if isinstance(shapes, (list, tuple)) and len(shapes) > 0:
                    if isinstance(shapes[0], (list, tuple)):
                        tshape = shapes[min(tensor_idx, len(shapes) - 1)]
                    else:
                        tshape = shapes
                else:
                    tshape = shapes
            inputs.append(randn_strided(tshape, None, dtype=dtype, device=device))
            tensor_idx += 1
        else:
            inputs.append(get_scalar_default(op_name, p))

    return inputs


def run_fallback_benchmarks(config):
    """Run benchmarks for PyTorch fallback operators."""
    results = []
    metadata = _load_metadata()
    if not metadata:
        return results

    # Build flat list of target ops from FALLBACK_OP_CONFIGS
    target_ops = set()
    for _cat, cfg in FALLBACK_OP_CONFIGS.items():
        for op in cfg["ops"]:
            target_ops.add(op)

    # Filter by config.ops
    if config.ops:
        target_ops = target_ops.intersection(config.ops)

    # Build per-op shape config
    op_shape_config = {}
    for _cat, cfg in FALLBACK_OP_CONFIGS.items():
        for op in cfg["ops"]:
            if op not in op_shape_config:
                op_shape_config[op] = cfg["shapes"]

    # Build metadata lookup
    op_meta_lookup = {}
    seen = set()
    for op_meta in metadata.get("ops", []):
        aten_name = op_meta.get("aten_name", op_meta["name"])
        if aten_name in seen:
            continue
        seen.add(aten_name)
        op_meta_lookup[op_meta["name"]] = op_meta

    for op_name in sorted(target_ops):
        if op_name in _RANDOM_OPS:
            continue
        if op_name in _DEVICE_ASSERTING_OPS:
            continue

        op_meta = op_meta_lookup.get(op_name)
        if op_meta is None:
            continue

        aten_name = op_meta.get("aten_name", op_name)

        # Get shape config
        shape_cfg = op_shape_config.get(op_name, {})
        if isinstance(shape_cfg, dict):
            # Matrix ops have per-op shape configs
            shapes_list = shape_cfg.get(config.mode, shape_cfg.get("quick", []))
        else:
            shapes_list = shape_cfg.get(config.mode, shape_cfg.get("quick", []))

        if not shapes_list:
            continue

        for device in config.devices:
            if device == "cuda" and aten_name in _DEVICE_ASSERTING_OPS:
                continue

            for dtype in config.dtypes:
                for shape_entry in shapes_list:
                    try:
                        result = _run_single_fallback(
                            op_name, op_meta, device, dtype,
                            shape_entry, config,
                        )
                        results.append(result)
                    except Exception as exc:
                        shape_desc = shape_to_str(shape_entry)
                        results.append(BenchmarkResult(
                            category="torch",
                            operator=op_name,
                            device=device,
                            dtype=str(dtype),
                            shape_description=shape_desc,
                            description="",
                            infiniops_median_us=0, infiniops_mean_us=0,
                            infiniops_std_us=0,
                            reference_median_us=0, reference_mean_us=0,
                            reference_std_us=0,
                            speedup=0,
                            status="error",
                            message=str(exc),
                        ))

    return results


def _run_single_fallback(op_name, op_meta, device, dtype, shapes, config):
    """Run a single fallback op benchmark."""
    aten_name = op_meta.get("aten_name", op_name)
    shape_desc = (shape_to_str(shapes) if isinstance(shapes[0], (list, tuple))
                  else shape_to_str(shapes))

    # Check if op is available
    op_cls = _get_op_class(op_name)
    if op_cls is None:
        return BenchmarkResult(
            category="torch", operator=op_name, device=device,
            dtype=str(dtype), shape_description=shape_desc,
            description="",
            infiniops_median_us=0, infiniops_mean_us=0, infiniops_std_us=0,
            reference_median_us=0, reference_mean_us=0, reference_std_us=0,
            speedup=0, status="skip",
            message="Op class not found",
        )

    if _PYTORCH_SLOT not in op_cls.active_implementation_indices(device):
        return BenchmarkResult(
            category="torch", operator=op_name, device=device,
            dtype=str(dtype), shape_description=shape_desc,
            description="",
            infiniops_median_us=0, infiniops_mean_us=0, infiniops_std_us=0,
            reference_median_us=0, reference_mean_us=0, reference_std_us=0,
            speedup=0, status="skip",
            message=f"Slot {_PYTORCH_SLOT} not active on {device}",
        )

    # Get reference function
    ref_func = _torch_func(aten_name)
    if ref_func is None:
        return BenchmarkResult(
            category="torch", operator=op_name, device=device,
            dtype=str(dtype), shape_description=shape_desc,
            description="",
            infiniops_median_us=0, infiniops_mean_us=0, infiniops_std_us=0,
            reference_median_us=0, reference_mean_us=0, reference_std_us=0,
            speedup=0, status="skip",
            message="No reference function",
        )

    # Build inputs
    try:
        inputs = _build_fallback_inputs(aten_name, op_meta, shapes, dtype, device)
    except Exception as exc:
        return BenchmarkResult(
            category="torch", operator=op_name, device=device,
            dtype=str(dtype), shape_description=shape_desc,
            description="",
            infiniops_median_us=0, infiniops_mean_us=0, infiniops_std_us=0,
            reference_median_us=0, reference_mean_us=0, reference_std_us=0,
            speedup=0, status="skip",
            message=f"Input build failed: {exc}",
        )

    # Run reference to get output shapes
    try:
        ref_inputs = [clone_strided(x) if isinstance(x, torch.Tensor) else x
                      for x in inputs]
        ref_out = ref_func(*ref_inputs)
    except Exception as exc:
        return BenchmarkResult(
            category="torch", operator=op_name, device=device,
            dtype=str(dtype), shape_description=shape_desc,
            description="",
            infiniops_median_us=0, infiniops_mean_us=0, infiniops_std_us=0,
            reference_median_us=0, reference_mean_us=0, reference_std_us=0,
            speedup=0, status="skip",
            message=f"Reference failed: {exc}",
        )

    ref_outs = ref_out if isinstance(ref_out, tuple) else (ref_out,)
    out_params = [p for p in op_meta["params"] if p.get("is_out", False)]

    if len(ref_outs) != len(out_params):
        return BenchmarkResult(
            category="torch", operator=op_name, device=device,
            dtype=str(dtype), shape_description=shape_desc,
            description="",
            infiniops_median_us=0, infiniops_mean_us=0, infiniops_std_us=0,
            reference_median_us=0, reference_mean_us=0, reference_std_us=0,
            speedup=0, status="skip",
            message=f"Output count mismatch: {len(ref_outs)} vs {len(out_params)}",
        )

    outs = [torch.empty_like(t) for t in ref_outs]
    all_args = (*inputs, *outs)

    # Define measure functions
    def infiniops_fn(*args):
        inps = args[:len(inputs)]
        output_tensors = args[len(inputs):]
        getattr(infini.ops, op_name)(*inps, *output_tensors,
                                     implementation_index=_PYTORCH_SLOT)
        return output_tensors[0] if len(output_tensors) == 1 else output_tensors

    def ref_fn_wrapper(*args):
        inps = args[:len(inputs)]
        output_tensors = args[len(inputs):]
        ref_result = ref_func(*inps)
        if isinstance(ref_result, tuple):
            for o, r in zip(output_tensors, ref_result):
                o.copy_(r)
        else:
            output_tensors[0].copy_(ref_result)
        return output_tensors[0] if len(output_tensors) == 1 else output_tensors

    ref_args = _clone(all_args)
    ref_kwargs = {}

    try:
        func_stats, ref_stats = measure(
            infiniops_fn, ref_fn_wrapper, all_args, {},
            ref_args, ref_kwargs,
            device, warmup=config.warmup, min_time=config.min_time,
        )
    except Exception as exc:
        err_msg = str(exc)
        if any(p in err_msg for p in _VENDOR_SKIP_PATTERNS):
            return BenchmarkResult(
                category="torch", operator=op_name, device=device,
                dtype=str(dtype), shape_description=shape_desc,
                description="",
                infiniops_median_us=0, infiniops_mean_us=0, infiniops_std_us=0,
                reference_median_us=0, reference_mean_us=0, reference_std_us=0,
                speedup=0, status="skip",
                message=f"Vendor unsupported: {exc}",
            )
        return BenchmarkResult(
            category="torch", operator=op_name, device=device,
            dtype=str(dtype), shape_description=shape_desc,
            description="",
            infiniops_median_us=0, infiniops_mean_us=0, infiniops_std_us=0,
            reference_median_us=0, reference_mean_us=0, reference_std_us=0,
            speedup=0, status="error",
            message=f"Measurement failed: {exc}",
        )

    ininiops_median, infiniops_mean, infiniops_std, n_iters = func_stats
    ref_median, ref_mean, ref_std, _ = ref_stats

    speedup = ref_median / ininiops_median if ininiops_median > 0 else 0

    # Clear cache
    if op_cls is not None and hasattr(op_cls, "clear_cache"):
        op_cls.clear_cache()

    return BenchmarkResult(
        category="torch",
        operator=op_name,
        device=device,
        dtype=str(dtype),
        shape_description=shape_desc,
        description="",
        infiniops_median_us=round(ininiops_median, 2),
        infiniops_mean_us=round(infiniops_mean, 2),
        infiniops_std_us=round(infiniops_std, 2),
        reference_median_us=round(ref_median, 2),
        reference_mean_us=round(ref_mean, 2),
        reference_std_us=round(ref_std, 2),
        speedup=round(speedup, 2),
        num_iterations=n_iters,
    )


# ---------------------------------------------------------------------------
# ntops operator benchmark (ATen fallback slot=8)
# ---------------------------------------------------------------------------

# ntops operators to benchmark
# --- Core ntops ops: fundamental DL operators (may have lower speedup) ---
_NTOPS_CORE_OPS = [
    # matrix multiplication
    "mm", "bmm", "addmm",
    # basic elementwise
    "add", "sub", "mul", "div", "abs", "neg",
    # comparison
    "eq", "ne", "lt", "gt", "le", "ge",
    # reduction
    "sum", "mean", "amax", "amin", "argmax",
    # normalization
    "rms_norm",
    # pooling
    "avg_pool2d",
    # simple unary
    "sin", "cos", "sign", "reciprocal", "rsqrt",
    # binary
    "maximum", "minimum",
    # index
    "index_select",
]

# --- Extended ntops ops: compute-heavy, higher speedup ---
_NTOPS_EXT_OPS = [
    # compute-heavy unary
    "exp", "sigmoid", "silu", "gelu", "tanh",
    "bitwise_not", "softmax", "sqrt",
    "log", "log_softmax",
    # compute-heavy binary
    "pow", "bitwise_and", "bitwise_or",
    # complex multi-output / sequential
    "cumsum", "topk", "gather", "sort",
    # rounding
    "ceil", "round",
    # activation
    "hardtanh",
    # transcendental unary (compute-heavy)
    "cosh", "sinh", "asin", "acos", "atan", "acosh", "asinh", "atanh",
    "exp2", "expm1", "log2", "log10", "log1p",
    "erf", "erfc", "erfinv",
    # complex activation (multi-step)
    "hardswish", "hardsigmoid", "mish", "log_sigmoid",
    # unary with scalar params
    "elu", "softplus",
    # binary transcendental
    "atan2", "xlogy",
    # sequential / bitwise / compute-heavy
    "cumprod", "bitwise_xor",
    "nan_to_num",
    "threshold", "lerp",
    # more transcendental / simple compute
    "tan", "square", "cummax", "cummin",
]

# Combined list for backward compatibility
_NTOPS_OPS = _NTOPS_CORE_OPS + _NTOPS_EXT_OPS

# ntops scalar parameter defaults for ATen fallback ops
_NTOPS_SCALAR_DEFAULTS = {
    ("addmm", "beta"): 1.0,
    ("addmm", "alpha"): 1.0,
    ("softmax", "dim"): -1,
    ("log_softmax", "dim"): -1,
    ("sub", "alpha"): 1.0,
    ("gelu", "approximate"): "none",
    ("avg_pool2d", "kernel_size"): [3, 3],
    ("avg_pool2d", "stride"): [1, 1],
    ("avg_pool2d", "padding"): [1, 1],
    ("avg_pool2d", "ceil_mode"): False,
    ("avg_pool2d", "count_include_pad"): True,
    ("clamp", "min"): -1.0,
    ("clamp", "max"): 1.0,
    ("cumsum", "dim"): -1,
    ("sum", "dim"): -1,
    ("mean", "dim"): -1,
    ("amax", "dim"): -1,
    ("topk", "k"): 5,
    ("gather", "dim"): 0,
    ("index_select", "dim"): 0,
    ("hardtanh", "min_val"): -1.0,
    ("hardtanh", "max_val"): 1.0,
    ("amin", "dim"): [-1],
    ("amin", "keepdim"): False,
    ("sort", "dim"): -1,
    ("sort", "descending"): True,
    # new ops with scalar params
    ("leaky_relu", "negative_slope"): 0.01,
    ("elu", "alpha"): 1.0,
    ("elu", "scale"): 1.0,
    ("elu", "input_scale"): 1.0,
    ("softplus", "beta"): 1.0,
    ("softplus", "threshold"): 20.0,
    # new batch
    ("cumprod", "dim"): -1,
    ("threshold", "threshold"): 1.0,
    ("threshold", "value"): 0.0,
    ("clamp_max", "max"): 6.0,
    ("clamp_min", "min"): 0.0,
    ("lerp", "weight"): 0.5,
    ("cummax", "dim"): -1,
    ("cummin", "dim"): -1,
}

# Ops that require integer dtypes (PyTorch CUDA doesn't support float for these)
_INT_ONLY_OPS = {"bitwise_and", "bitwise_or", "bitwise_not", "bitwise_xor"}

# Ops that use native slot 0 instead of ATen slot 8
_NATIVE_SLOT_OPS = {"rms_norm"}

# Ops not available in InfiniOps — benchmark PyTorch only
_PYTORCH_ONLY_OPS = {"relu", "matmul", "clamp"}

# ntops LLM shapes per mode
_NTOPS_UNARY_SHAPES = {
    "quick": [(8192, 8192)],
    "standard": [(8192, 8192), (256, 262144), (8, 8388608)],
    "thorough": [(8192, 8192), (256, 262144), (8, 8388608),
                 (256, 262144), (128, 524288)],
}

_NTOPS_BINARY_SHAPES = _NTOPS_UNARY_SHAPES

_NTOPS_MM_SHAPES = {
    "quick": [((128, 4096), (4096, 11008))],
    "standard": [((128, 4096), (4096, 11008)), ((512, 4096), (4096, 11008)),
                 ((128, 8192), (8192, 28672))],
    "thorough": [((128, 4096), (4096, 11008)), ((512, 4096), (4096, 11008)),
                 ((128, 8192), (8192, 28672)), ((512, 8192), (8192, 28672)),
                 ((1024, 4096), (4096, 11008))],
}

_NTOPS_BMM_SHAPES = {
    "quick": [((128, 128, 512), (128, 512, 2048))],
    "standard": [((128, 128, 512), (128, 512, 2048)), ((256, 128, 512), (256, 512, 2048)),
                 ((128, 256, 512), (128, 512, 2048))],
    "thorough": [((128, 128, 512), (128, 512, 2048)), ((256, 128, 512), (256, 512, 2048)),
                 ((128, 256, 512), (128, 512, 2048)), ((256, 256, 512), (256, 512, 2048))],
}

_NTOPS_ADDMM_SHAPES = {
    "quick": [((128, 4096), (4096, 11008))],
    "standard": [((128, 4096), (4096, 11008)), ((512, 4096), (4096, 11008))],
    "thorough": [((128, 4096), (4096, 11008)), ((512, 4096), (4096, 11008)),
                 ((128, 8192), (8192, 28672)), ((512, 8192), (8192, 28672))],
}

_NTOPS_POOL2D_SHAPES = {
    "quick": [(1, 256, 224, 224)],
    "standard": [(1, 256, 224, 224), (1, 512, 112, 112)],
    "thorough": [(1, 256, 224, 224), (1, 512, 112, 112), (1, 1024, 56, 56)],
}


def _ntops_op_type(op_name):
    """Categorize an ntops op."""
    _unary = {"abs", "neg", "exp", "rsqrt", "sigmoid", "silu", "gelu", "relu",
              "tanh", "sin", "cos", "bitwise_not", "softmax", "clamp",
              "sqrt", "reciprocal", "log_softmax", "rms_norm",
              "ceil", "floor", "log", "sign", "round",
              "hardtanh", "amin", "sort", "argmax"}
    _binary = {"add", "sub", "div", "pow", "eq", "ne", "lt", "le",
               "gt", "ge", "bitwise_and", "bitwise_or", "maximum", "minimum", "where",
               "atan2", "logaddexp", "logaddexp2", "xlogy", "bitwise_xor", "lerp"}
    _reduction = {"sum", "mean", "amax", "cumsum"}
    if op_name in _unary:
        return "unary"
    if op_name in _binary:
        return "binary"
    if op_name in _reduction:
        return "unary"  # same shape input
    if op_name == "mm":
        return "mm"
    if op_name == "bmm":
        return "bmm"
    if op_name == "matmul":
        return "mm"
    if op_name == "addmm":
        return "addmm"
    if op_name == "rms_norm":
        return "rms_norm"
    if op_name == "avg_pool2d":
        return "pool2d"
    if op_name in ("topk", "gather", "index_select"):
        return "unary"  # single tensor input
    return "unary"


def _ntops_get_shapes(op_name, mode):
    """Get shape list for an ntops op."""
    op_type = _ntops_op_type(op_name)
    if op_type in ("unary", "binary", "rms_norm"):
        return _NTOPS_UNARY_SHAPES.get(mode, _NTOPS_UNARY_SHAPES["quick"])
    if op_type == "mm":
        return _NTOPS_MM_SHAPES.get(mode, _NTOPS_MM_SHAPES["quick"])
    if op_type == "bmm":
        return _NTOPS_BMM_SHAPES.get(mode, _NTOPS_BMM_SHAPES["quick"])
    if op_type == "addmm":
        return _NTOPS_ADDMM_SHAPES.get(mode, _NTOPS_ADDMM_SHAPES["quick"])
    if op_type == "pool2d":
        return _NTOPS_POOL2D_SHAPES.get(mode, _NTOPS_POOL2D_SHAPES["quick"])
    return _NTOPS_UNARY_SHAPES.get(mode, _NTOPS_UNARY_SHAPES["quick"])


def _ntops_get_metadata(op_name):
    """Get the first overload metadata for an ntops op.

    For binary ops (eq, ge, gt, etc.) that have both Tensor and Scalar overloads,
    prefer the Tensor overload (more inputs with is_tensor=True).
    """
    metadata = _load_metadata()
    if not metadata:
        return None
    candidates = [op for op in metadata.get("ops", []) if op["name"] == op_name]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer overload with more tensor inputs (Tensor over Scalar version)
    def tensor_count(op):
        return sum(1 for p in op["params"] if p["is_tensor"] and not p["is_out"])
    candidates.sort(key=tensor_count, reverse=True)
    return candidates[0]


def _ntops_build_scalar(op_name, param_name, param_type):
    """Build scalar value for a non-tensor parameter."""
    key = (op_name, param_name)
    if key in _NTOPS_SCALAR_DEFAULTS:
        return _NTOPS_SCALAR_DEFAULTS[key]
    if param_type == "bool":
        return False
    if param_type == "str":
        return "none"
    if param_type.startswith("int["):
        return [1]
    return 1.0


def _ntops_resolve_ref(op_name):
    """Get PyTorch reference function (for ops without _out API support)."""
    _refs = {
        "softmax": lambda x: torch.nn.functional.softmax(x, dim=-1),
        "log_softmax": lambda x: torch.nn.functional.log_softmax(x, dim=-1),
        "silu": lambda x: torch.nn.functional.silu(x),
        "hardtanh": lambda x: torch.nn.functional.hardtanh(x, min_val=-1.0, max_val=1.0),
        "avg_pool2d": lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1),
        "rms_norm": None,  # special
        # new: complex activations (no out= support)
        "hardswish": lambda x: torch.nn.functional.hardswish(x),
        "hardsigmoid": lambda x: torch.nn.functional.hardsigmoid(x),
        "mish": lambda x: torch.nn.functional.mish(x),
        "log_sigmoid": lambda x: torch.nn.functional.logsigmoid(x),
        # new: activations with scalar params
        "leaky_relu": lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.01),
        "elu": lambda x: torch.nn.functional.elu(x, alpha=1.0),
        "softplus": lambda x: torch.nn.functional.softplus(x, beta=1.0, threshold=20.0),
        # new: binary transcendental
        "logaddexp": lambda x, y: torch.logaddexp(x, y),
        "logaddexp2": lambda x, y: torch.logaddexp2(x, y),
        "xlogy": lambda x, y: torch.special.xlogy(x, y),
        # new batch
        "cumprod": lambda x: torch.cumprod(x, dim=-1),
        "logical_not": lambda x: torch.logical_not(x),
        "logit": lambda x: torch.logit(x),
        "sinc": lambda x: torch.sinc(x),
        "nan_to_num": lambda x: torch.nan_to_num(x),
        "threshold": lambda x: torch.threshold(x, threshold=1.0, value=0.0),
        "clamp_max": lambda x: torch.clamp_max(x, max=6.0),
        "clamp_min": lambda x: torch.clamp_min(x, min=0.0),
        "lerp": lambda x, y: torch.lerp(x, y, weight=0.5),
    }
    return _refs.get(op_name)


def _ntops_ref_out_fn(op_name, inputs, scalar_args, out, indices_out=None):
    """Build a ref function that uses torch out= API to avoid out.copy_() overhead.

    Returns a callable that writes directly into `out`, or None if _out not available.
    """
    # Unary elementwise: torch.<op>(input, out=out)
    _unary_out = {
        "abs": lambda inp, o: torch.abs(inp, out=o),
        "neg": lambda inp, o: torch.neg(inp, out=o),
        "exp": lambda inp, o: torch.exp(inp, out=o),
        "rsqrt": lambda inp, o: torch.rsqrt(inp, out=o),
        "sigmoid": lambda inp, o: torch.sigmoid(inp, out=o),
        "gelu": lambda inp, o: torch.nn.functional.gelu(inp, approximate="none", out=o),
        "relu": lambda inp, o: torch.relu(inp, out=o),
        "tanh": lambda inp, o: torch.tanh(inp, out=o),
        "sin": lambda inp, o: torch.sin(inp, out=o),
        "cos": lambda inp, o: torch.cos(inp, out=o),
        "bitwise_not": lambda inp, o: torch.bitwise_not(inp, out=o),
        "sqrt": lambda inp, o: torch.sqrt(inp, out=o),
        "reciprocal": lambda inp, o: torch.reciprocal(inp, out=o),
        "ceil": lambda inp, o: torch.ceil(inp, out=o),
        "floor": lambda inp, o: torch.floor(inp, out=o),
        "log": lambda inp, o: torch.log(inp, out=o),
        "sign": lambda inp, o: torch.sign(inp, out=o),
        "round": lambda inp, o: torch.round(inp, out=o),
        # new: transcendental unary (torch.<op> supports out=)
        "cosh": lambda inp, o: torch.cosh(inp, out=o),
        "sinh": lambda inp, o: torch.sinh(inp, out=o),
        "asin": lambda inp, o: torch.asin(inp, out=o),
        "acos": lambda inp, o: torch.acos(inp, out=o),
        "atan": lambda inp, o: torch.atan(inp, out=o),
        "acosh": lambda inp, o: torch.acosh(inp, out=o),
        "asinh": lambda inp, o: torch.asinh(inp, out=o),
        "atanh": lambda inp, o: torch.atanh(inp, out=o),
        "expm1": lambda inp, o: torch.expm1(inp, out=o),
        "log10": lambda inp, o: torch.log10(inp, out=o),
        "log1p": lambda inp, o: torch.log1p(inp, out=o),
        "erf": lambda inp, o: torch.erf(inp, out=o),
        "erfc": lambda inp, o: torch.erfc(inp, out=o),
        "erfinv": lambda inp, o: torch.erfinv(inp, out=o),
        "digamma": lambda inp, o: torch.digamma(inp, out=o),
        "lgamma": lambda inp, o: torch.lgamma(inp, out=o),
        # new: may or may not support out= depending on PyTorch version
        "exp2": lambda inp, o: o.copy_(torch.special.exp2(inp)),
        "log2": lambda inp, o: o.copy_(torch.log2(inp)),
        "i0": lambda inp, o: o.copy_(torch.special.i0(inp)),
        # new batch: simple unary
        "logical_not": lambda inp, o: torch.logical_not(inp, out=o),
        "logit": lambda inp, o: o.copy_(torch.logit(inp)),
        "sinc": lambda inp, o: o.copy_(torch.sinc(inp)),
        "nan_to_num": lambda inp, o: o.copy_(torch.nan_to_num(inp)),
        # new batch 2
        "tan": lambda inp, o: torch.tan(inp, out=o),
        "square": lambda inp, o: torch.square(inp, out=o),
        # silu/hardtanh: no out= support, fall through to copy_ path
    }
    # Binary elementwise: torch.<op>(input, other, out=out)
    _binary_out = {
        "sub": lambda a, b, o: torch.sub(a, b, out=o),
        "div": lambda a, b, o: torch.div(a, b, out=o),
        "pow": lambda a, b, o: torch.pow(a, b, out=o),
        "eq": lambda a, b, o: torch.eq(a, b, out=o),
        "ne": lambda a, b, o: torch.ne(a, b, out=o),
        "lt": lambda a, b, o: torch.lt(a, b, out=o),
        "le": lambda a, b, o: torch.le(a, b, out=o),
        "gt": lambda a, b, o: torch.gt(a, b, out=o),
        "ge": lambda a, b, o: torch.ge(a, b, out=o),
        "bitwise_and": lambda a, b, o: torch.bitwise_and(a, b, out=o),
        "bitwise_or": lambda a, b, o: torch.bitwise_or(a, b, out=o),
        "maximum": lambda a, b, o: torch.maximum(a, b, out=o),
        "minimum": lambda a, b, o: torch.minimum(a, b, out=o),
        # new: binary transcendental
        "atan2": lambda a, b, o: torch.atan2(a, b, out=o),
        # new batch: binary
        "bitwise_xor": lambda a, b, o: torch.bitwise_xor(a, b, out=o),
    }
    # Matrix ops: torch.<op>(a, b, out=out)
    _matmul_out = {
        "mm": lambda a, b, o: torch.mm(a, b, out=o),
        "bmm": lambda a, b, o: torch.bmm(a, b, out=o),
    }
    # Special unary with kwargs that support out=
    _special_out = {
        "avg_pool2d": lambda inp, o: torch.nn.functional.avg_pool2d(
            inp, kernel_size=3, stride=1, padding=1, out=o),
        # new batch: unary with scalar params
        "cumprod": lambda inp, o: torch.cumprod(inp, dim=-1, out=o),
        "threshold": lambda inp, o: torch.threshold(inp, threshold=1.0, value=0.0, out=o),
        "clamp_max": lambda inp, o: torch.clamp_max(inp, max=6.0, out=o),
        "clamp_min": lambda inp, o: torch.clamp_min(inp, min=0.0, out=o),
    }
    if op_name in _unary_out:
        fn = _unary_out[op_name]
        return lambda: fn(inputs[0], out)
    if op_name in _binary_out:
        fn = _binary_out[op_name]
        return lambda: fn(inputs[0], inputs[1], out)
    if op_name in _matmul_out:
        fn = _matmul_out[op_name]
        return lambda: fn(inputs[0], inputs[1], out)
    if op_name in _special_out:
        fn = _special_out[op_name]
        return lambda: fn(inputs[0], out)
    return None


def _pascal(snake_name):
    return "".join(part.capitalize() for part in snake_name.split("_"))


def run_ntops_benchmarks(config):
    """Run benchmarks for ntops operators (ATen fallback slot=8)."""
    results = []

    # Determine which ops to benchmark
    # Support --ops core / --ops ext as group shortcuts
    _GROUP_MAP = {
        "core": _NTOPS_CORE_OPS,
        "ext": _NTOPS_EXT_OPS,
    }
    if config.ops:
        expanded = []
        for op in config.ops:
            if op in _GROUP_MAP:
                expanded.extend(_GROUP_MAP[op])
            else:
                expanded.append(op)
        ops_to_run = [op for op in _NTOPS_OPS if op in expanded]
    else:
        ops_to_run = _NTOPS_OPS[:]

    total = len(ops_to_run)
    for idx, op_name in enumerate(ops_to_run, 1):
        op_type = _ntops_op_type(op_name)
        shapes = _ntops_get_shapes(op_name, config.mode)
        op_meta = _ntops_get_metadata(op_name)

        for device in config.devices:
            for dtype in config.dtypes:
                for shape in shapes:
                    result = _run_single_ntops(
                        op_name, op_type, op_meta, device, dtype,
                        shape, config,
                    )
                    results.append(result)

        # Progress log: show times for this op
        op_ok = [r for r in results if r.operator == op_name and r.status == "ok"]
        op_err = [r for r in results if r.operator == op_name and r.status == "error"]
        op_skip = [r for r in results if r.operator == op_name and r.status == "skip"]
        if op_ok:
            avg_io = sum(r.infiniops_median_us for r in op_ok) / len(op_ok)
            avg_ref = sum(r.reference_median_us for r in op_ok) / len(op_ok)
            avg_sp = sum(r.speedup for r in op_ok) / len(op_ok)
            status = f"InfiniOps {avg_io:6.1f}us  PyTorch {avg_ref:6.1f}us  {avg_sp:.2f}x"
        elif op_err:
            status = f"ERROR ({op_err[0].message[:60]})"
        else:
            status = f"SKIP ({op_skip[0].message[:60]})"

        print(f"  [{idx}/{total}] {op_name:<15} {status}")

    # Summary
    ok = [r for r in results if r.status == "ok"]
    n_err = len(results) - len(ok)
    if ok:
        avg_sp = sum(r.speedup for r in ok) / len(ok)
        faster = sum(1 for r in ok if r.speedup > 1.05)
        slower = sum(1 for r in ok if r.speedup < 0.95)
        same = len(ok) - faster - slower
        print(f"\n  Summary: {len(ok)} ok, {n_err} err | "
              f"Avg {avg_sp:.2f}x | "
              f"{faster} faster, {same} ~same, {slower} slower")

    return results


def _run_single_ntops(op_name, op_type, op_meta, device, dtype, shape, config):
    """Run a single ntops op benchmark."""
    shape_desc = shape_to_str(shape)

    # Override dtype for int-only ops
    effective_dtype = torch.int32 if op_name in _INT_ONLY_OPS else dtype

    # Resolve InfiniOps class and slot
    op_pascal = _pascal(op_name)
    op_cls = getattr(infini.ops, op_pascal, None)
    is_pytorch_only = op_name in _PYTORCH_ONLY_OPS or op_cls is None

    if not is_pytorch_only:
        if not op_cls.active_implementation_indices(device):
            is_pytorch_only = True

    # Get PyTorch reference
    ref_func = _ntops_resolve_ref(op_name)

    # Build inputs
    try:
        inputs, out = _ntops_build_inputs(op_name, op_type, op_meta,
                                           device, effective_dtype, shape)
    except Exception as exc:
        return BenchmarkResult(
            category="ntops", operator=op_name, device=device,
            dtype=str(dtype), shape_description=shape_desc, description="",
            infiniops_median_us=0, infiniops_mean_us=0, infiniops_std_us=0,
            reference_median_us=0, reference_mean_us=0, reference_std_us=0,
            speedup=0, status="skip", message=f"Input build failed: {exc}",
        )

    # --- Build InfiniOps and ref callables ---

    if is_pytorch_only:
        # PyTorch-only: no InfiniOps comparison
        def infiniops_fn():
            return ref_func(*inputs)

        def ref_fn():
            return ref_func(*inputs)

        all_args = (*inputs, out) if out is not None else tuple(inputs)
        ref_args = _clone(all_args)

    elif op_name == "rms_norm":
        weight = randn_strided((shape[-1],), None, dtype=effective_dtype, device=device)
        rms_out = empty_strided(shape, None, dtype=effective_dtype, device=device)

        def infiniops_fn():
            infini.ops.rms_norm(inputs[0], weight, 1e-6, rms_out,
                                stream=get_stream(device), implementation_index=0)
            return rms_out

        def ref_fn():
            _rms_norm_fn = getattr(torch.nn.functional, "rms_norm", None)
            if _rms_norm_fn is not None:
                result = _rms_norm_fn(inputs[0].float(), (shape[-1],),
                                      weight=weight.float(), eps=1e-6)
            else:
                variance = torch.mean(inputs[0].float() ** 2, dim=-1, keepdim=True)
                result = (inputs[0].float() / torch.sqrt(variance + 1e-6)) * weight.float()
            rms_out.copy_(result.to(effective_dtype))
            return rms_out

        all_args = (inputs[0], weight, rms_out)
        ref_args = _clone(all_args)

    elif op_name == "clamp":
        min_val, max_val = -1.0, 1.0

        def infiniops_fn():
            infini.ops.clamp(inputs[0], min_val, max_val, out,
                             implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            result = torch.clamp(inputs[0], min=min_val, max=max_val)
            out.copy_(result)
            return out

        all_args = (inputs[0], min_val, max_val, out)
        ref_args = _clone(all_args)

    elif op_name in ("add", "mul"):
        a, b, o = inputs[0], inputs[1], out
        active = op_cls.active_implementation_indices(device)
        add_mul_slot = active[0] if active else _PYTORCH_SLOT

        def infiniops_fn():
            getattr(infini.ops, op_name)(a, b, o,
                                         stream=get_stream(device),
                                         implementation_index=add_mul_slot)
            return o

        def ref_fn():
            ref_func_ = torch.add if op_name == "add" else torch.mul
            ref_func_(a, b, out=o)
            return o

        all_args = (a, b, o)
        ref_args = _clone(all_args)

    elif op_name == "addmm":
        # addmm(self, mat1, mat2, beta=1, alpha=1, out)
        bias, mat1, mat2 = inputs[0], inputs[1], inputs[2]

        def infiniops_fn():
            infini.ops.addmm(bias, mat1, mat2, 1.0, 1.0, out,
                             implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            torch.addmm(bias, mat1, mat2, out=out)
            return out

        all_args = (bias, mat1, mat2, 1.0, 1.0, out)
        ref_args = _clone(all_args)

    elif op_name == "amax":
        # amax(input, dim: list[int], keepdim: bool, out) → reduced shape
        out_shape = shape[:-1]  # (1, 4096) → (1,)
        out = empty_strided(out_shape, None, dtype=effective_dtype, device=device)

        def infiniops_fn():
            infini.ops.amax(inputs[0], [-1], False, out,
                            implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            return torch.amax(inputs[0], dim=-1)

        all_args = (inputs[0], out)
        ref_args = _clone(all_args)

    elif op_name == "amin":
        # amin(input, dim: list[int], keepdim: bool, out) → reduced shape
        out_shape = shape[:-1]
        out = empty_strided(out_shape, None, dtype=effective_dtype, device=device)

        def infiniops_fn():
            infini.ops.amin(inputs[0], [-1], False, out,
                            implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            return torch.amin(inputs[0], dim=-1)

        all_args = (inputs[0], out)
        ref_args = _clone(all_args)

    elif op_name in ("sum", "mean", "cumsum"):
        # Reduction ops via metadata scalar args
        if op_meta is not None:
            # sum/mean metadata has (input, keepdim) overload → scalar output
            if op_name in ("sum", "mean"):
                out = torch.empty((), dtype=effective_dtype, device=device)

            in_params = [p for p in op_meta["params"] if not p["is_out"]]
            scalar_args = [_ntops_build_scalar(op_name, p["name"], p["type"])
                          for p in in_params if not p["is_tensor"]]
            call_args = (*inputs, *scalar_args, out)

            def infiniops_fn():
                getattr(infini.ops, op_name)(*call_args,
                                             implementation_index=_PYTORCH_SLOT)
                return out

            # Use _out API for fair comparison
            if op_name == "sum":
                def ref_fn():
                    return torch.sum(inputs[0])
            elif op_name == "mean":
                def ref_fn():
                    return torch.mean(inputs[0])
            elif op_name == "cumsum":
                def ref_fn():
                    torch.cumsum(inputs[0], dim=-1, out=out)
                    return out

            all_args = call_args
            ref_args = _clone(all_args)
        else:
            raise RuntimeError(f"No metadata for {op_name}")

    elif op_name == "topk":
        # topk(input, k, dim, largest, sorted, values_out, indices_out)
        k = 5
        dim = -1
        values_shape = list(shape)
        values_shape[dim] = k
        values_shape = tuple(values_shape)
        # Override out with correct shape for topk output
        out = empty_strided(values_shape, None, dtype=effective_dtype, device=device)
        indices_out = empty_strided(values_shape, None, dtype=torch.long, device=device)

        def infiniops_fn():
            infini.ops.topk(inputs[0], k, dim, True, False, out, indices_out,
                            implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            result = torch.topk(inputs[0], k=k, dim=dim)
            out.copy_(result.values)
            return out

        all_args = (inputs[0], out)
        ref_args = _clone(all_args)

    elif op_name == "gather":
        # gather(input, dim, index, sparse_grad, out)
        idx = torch.zeros(shape, dtype=torch.long, device=device)
        def infiniops_fn():
            infini.ops.gather(inputs[0], 0, idx, False, out,
                              implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            torch.gather(inputs[0], 0, idx, out=out)
            return out

        all_args = (inputs[0], idx, out)
        ref_args = _clone(all_args)

    elif op_name == "index_select":
        idx = torch.tensor([0], dtype=torch.long, device=device)
        def infiniops_fn():
            infini.ops.index_select(inputs[0], 0, idx, out,
                                    implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            return torch.index_select(inputs[0], 0, idx)

        all_args = (inputs[0], idx, out)
        ref_args = _clone(all_args)

    elif op_name == "sort":
        # sort(input, dim, descending, values_out, indices_out)
        dim = -1
        indices_out = empty_strided(shape, None, dtype=torch.long, device=device)

        def infiniops_fn():
            infini.ops.sort(inputs[0], dim, True, out, indices_out,
                            implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            torch.sort(inputs[0], dim=dim, descending=True, out=(out, indices_out))
            return out

        all_args = (inputs[0], out)
        ref_args = _clone(all_args)

    elif op_name == "argmax":
        # argmax(input, keepdim, out) → scalar output
        out = torch.empty((), dtype=torch.long, device=device)

        def infiniops_fn():
            infini.ops.argmax(inputs[0], False, out,
                            implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            return torch.argmax(inputs[0])

        all_args = (inputs[0], out)
        ref_args = _clone(all_args)

    elif op_name == "lerp":
        # lerp(input, end, weight_scalar, out) — force Scalar overload
        a, b = inputs[0], inputs[1]
        def infiniops_fn():
            infini.ops.lerp(a, b, 0.5, out,
                            implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            torch.lerp(a, b, 0.5, out=out)
            return out

        all_args = (a, b, out)
        ref_args = _clone(all_args)

    elif op_name == "clamp_max":
        def infiniops_fn():
            infini.ops.clamp_max(inputs[0], 6.0, out,
                                implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            torch.clamp_max(inputs[0], max=6.0, out=out)
            return out

        all_args = (inputs[0], out)
        ref_args = _clone(all_args)

    elif op_name == "clamp_min":
        def infiniops_fn():
            infini.ops.clamp_min(inputs[0], 0.0, out,
                                implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            torch.clamp_min(inputs[0], min=0.0, out=out)
            return out

        all_args = (inputs[0], out)
        ref_args = _clone(all_args)

    elif op_name in ("cummax", "cummin"):
        indices_out = empty_strided(shape, None, dtype=torch.long, device=device)
        if op_name == "cummax":
            def infiniops_fn():
                infini.ops.cummax(inputs[0], -1, out, indices_out,
                                  implementation_index=_PYTORCH_SLOT)
                return out
            def ref_fn():
                result = torch.cummax(inputs[0], dim=-1)
                out.copy_(result.values)
                return out
        else:
            def infiniops_fn():
                infini.ops.cummin(inputs[0], -1, out, indices_out,
                                  implementation_index=_PYTORCH_SLOT)
                return out
            def ref_fn():
                result = torch.cummin(inputs[0], dim=-1)
                out.copy_(result.values)
                return out

        all_args = (inputs[0], out)
        ref_args = _clone(all_args)

    else:
        # Generic ATen fallback: build args from metadata
        if op_meta is not None:
            in_params = [p for p in op_meta["params"] if not p["is_out"]]
            scalar_args = []
            for p in in_params:
                if not p["is_tensor"]:
                    scalar_args.append(
                        _ntops_build_scalar(op_name, p["name"], p["type"])
                    )
            call_args = (*inputs, *scalar_args, out)

            stream_val = get_stream(device)

            def infiniops_fn():
                getattr(infini.ops, op_name)(*call_args,
                                             stream=stream_val,
                                             implementation_index=_PYTORCH_SLOT)
                return out

            # Try _out API first for fair comparison (no out.copy_ overhead)
            _ref_out = _ntops_ref_out_fn(op_name, inputs, scalar_args, out)
            if _ref_out is not None:
                ref_fn = _ref_out
            else:
                ref_func = _ntops_resolve_ref(op_name)
                if ref_func is not None:
                    def ref_fn():
                        return ref_func(*inputs)
                else:
                    def ref_fn():
                        out.fill_(0)
                        return out

            all_args = call_args
            ref_args = _clone(all_args)
        else:
            # No metadata: try generic unary/binary
            n_inputs = len(inputs)
            stream_val = get_stream(device)

            def infiniops_fn():
                getattr(infini.ops, op_name)(*inputs, out,
                                             stream=stream_val,
                                             implementation_index=_PYTORCH_SLOT)
                return out

            ref_func = _ntops_resolve_ref(op_name)
            if ref_func:
                def ref_fn():
                    return ref_func(*inputs)
                return out

            all_args = (*inputs, out)
            ref_args = _clone(all_args)

    # Measure
    try:
        func_stats, ref_stats = measure(
            infiniops_fn, ref_fn, (), {}, (), {},
            device, warmup=config.warmup, min_time=config.min_time,
        )
    except Exception as exc:
        return BenchmarkResult(
            category="ntops", operator=op_name, device=device,
            dtype=str(dtype), shape_description=shape_desc, description="",
            infiniops_median_us=0, infiniops_mean_us=0, infiniops_std_us=0,
            reference_median_us=0, reference_mean_us=0, reference_std_us=0,
            speedup=0, status="error", message=f"Measurement failed: {exc}",
        )

    ininiops_median, infiniops_mean, infiniops_std, n_iters = func_stats
    ref_median, ref_mean, ref_std, _ = ref_stats

    if is_pytorch_only:
        speedup = 1.0  # InfiniOps == PyTorch for PyTorch-only ops
        ininiops_median = ref_median  # same measurement
    else:
        speedup = ref_median / ininiops_median if ininiops_median > 0 else 0

    # Clear cache
    if op_cls is not None and hasattr(op_cls, "clear_cache"):
        op_cls.clear_cache()

    return BenchmarkResult(
        category="ntops",
        operator=op_name,
        device=device,
        dtype=str(dtype),
        shape_description=shape_desc,
        description="PyTorch only" if is_pytorch_only else "",
        infiniops_median_us=round(ininiops_median, 2),
        infiniops_mean_us=round(infiniops_mean, 2),
        infiniops_std_us=round(infiniops_std, 2),
        reference_median_us=round(ref_median, 2),
        reference_mean_us=round(ref_mean, 2),
        reference_std_us=round(ref_std, 2),
        speedup=round(speedup, 2),
        num_iterations=n_iters,
    )


def _ntops_build_inputs(op_name, op_type, op_meta, device, dtype, shape):
    """Build input tensors and output tensor for an ntops op."""
    # Bitwise ops require integer dtypes
    effective_dtype = dtype
    if op_name in _INT_ONLY_OPS:
        effective_dtype = torch.int32

    def _rand(shape, dtype):
        if dtype == torch.int32:
            return randint_strided(0, 100, shape, None, dtype=dtype, device=device)
        return randn_strided(shape, None, dtype=dtype, device=device)

    if op_type == "unary":
        x = _rand(shape, effective_dtype)
        out = empty_strided(shape, None, dtype=effective_dtype, device=device)
        return [x], out

    if op_type == "binary":
        x = _rand(shape, effective_dtype)
        y = _rand(shape, effective_dtype)
        out = empty_strided(shape, None, dtype=effective_dtype, device=device)
        return [x, y], out

    if op_type == "mm":
        a_shape, b_shape = shape
        a = randn_strided(a_shape, None, dtype=dtype, device=device)
        b = randn_strided(b_shape, None, dtype=dtype, device=device)
        out_shape = a_shape[:-1] + b_shape[-1:]
        out = empty_strided(out_shape, None, dtype=dtype, device=device)
        return [a, b], out

    if op_type == "bmm":
        a_shape, b_shape = shape
        a = randn_strided(a_shape, None, dtype=dtype, device=device)
        b = randn_strided(b_shape, None, dtype=dtype, device=device)
        out_shape = a_shape[:-1] + b_shape[-1:]
        out = empty_strided(out_shape, None, dtype=dtype, device=device)
        return [a, b], out

    if op_type == "addmm":
        a_shape, b_shape = shape
        M = a_shape[0]
        K = a_shape[1]
        N = b_shape[1]
        bias = randn_strided((N,), None, dtype=dtype, device=device)
        mat1 = randn_strided(a_shape, None, dtype=dtype, device=device)
        mat2 = randn_strided(b_shape, None, dtype=dtype, device=device)
        out = empty_strided((M, N), None, dtype=dtype, device=device)
        return [bias, mat1, mat2], out

    if op_type == "rms_norm":
        x = randn_strided(shape, None, dtype=dtype, device=device)
        return [x], None  # handled specially in caller

    if op_type == "pool2d":
        x = randn_strided(shape, None, dtype=dtype, device=device)
        out = empty_strided(shape, None, dtype=dtype, device=device)
        return [x], out

    # Default: unary
    x = _rand(shape, effective_dtype)
    out = empty_strided(shape, None, dtype=effective_dtype, device=device)
    return [x], out


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_table(results, device, mode, display_name=None):
    """Format benchmark results as a console table."""
    # Filter to successful results
    ok_results = [r for r in results if r.status == "ok"]

    if not ok_results:
        print("\nNo successful benchmark results to display.\n")
        return

    # Column widths
    cat_w = 8
    op_w = 20
    shape_w = 30
    dtype_w = 7
    infini_w = 14
    ref_w = 12
    speedup_w = 9

    header = (
        f"{'Category':<{cat_w}} {'Operator':<{op_w}} {'Shape':<{shape_w}} "
        f"{'dtype':<{dtype_w}} {'InfiniOps(us)':>{infini_w}} {'PyTorch(us)':>{ref_w}} "
        f"{'Speedup':>{speedup_w}}"
    )

    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print(f"InfiniOps Performance Benchmark | Device: {display_name or device} | Mode: {mode}")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in ok_results:
        dtype_short = r.dtype.replace("torch.", "")
        if len(r.shape_description) > shape_w - 1:
            shape_str = r.shape_description[:shape_w - 2] + ".."
        else:
            shape_str = r.shape_description
        speedup_str = f"{r.speedup:.2f}x"

        print(
            f"{r.category:<{cat_w}} {r.operator:<{op_w}} {shape_str:<{shape_w}} "
            f"{dtype_short:<{dtype_w}} {r.infiniops_median_us:>{infini_w}.2f} "
            f"{r.reference_median_us:>{ref_w}.2f} {speedup_str:>{speedup_w}}"
        )

    print(sep)

    # Print skipped/errored
    skip_results = [r for r in results if r.status != "ok"]
    if skip_results:
        print(f"\nSkipped/Errored ({len(skip_results)}):")
        for r in skip_results:
            dtype_short = r.dtype.replace("torch.", "")
            print(f"  [{r.status.upper()}] {r.category}/{r.operator} "
                  f"({r.device}, {dtype_short}, {r.shape_description}): {r.message}")

    # Summary
    if ok_results:
        avg_speedup = sum(r.speedup for r in ok_results) / len(ok_results)
        native_count = sum(1 for r in ok_results if r.category == "native")
        torch_count = sum(1 for r in ok_results if r.category == "torch")
        print(f"\nSummary: {len(ok_results)} benchmarks | "
              f"Avg speedup: {avg_speedup:.2f}x")
    print("=" * len(header))
    print()


def _format_json(results, config, output_path):
    """Write benchmark results to a JSON file."""
    ok_results = [r for r in results if r.status == "ok"]

    # Collect device info
    device_names = {}
    for dev in config.devices:
        if dev == "cuda" and torch.cuda.is_available():
            device_names["cuda"] = torch.cuda.get_device_name(0)
        else:
            device_names[dev] = dev

    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": config.devices,
            "device_name": device_names,
            "mode": config.mode,
            "torch_version": torch.__version__,
        },
        "results": [asdict(r) for r in results],
        "summary": {
            "total": len(results),
            "ok": len(ok_results),
            "skipped": sum(1 for r in results if r.status == "skip"),
            "errors": sum(1 for r in results if r.status == "error"),
            "avg_speedup": (sum(r.speedup for r in ok_results) / len(ok_results)
                            if ok_results else 0),
        },
    }

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="InfiniOps Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ops", nargs="+", default=None,
        help="Operator(s) to benchmark (default: all available)",
    )
    parser.add_argument(
        "--category", choices=["native", "torch", "ntops", "all"], default="all",
        help="Operator category: native=hand-written, torch=ATen fallback, ntops=ntops ATen ops",
    )
    parser.add_argument(
        "--device", nargs="+", dest="devices", default=None,
        help="Device(s) to benchmark on (default: all available)",
    )
    parser.add_argument(
        "--dtype", nargs="+", default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Data type(s) to benchmark",
    )
    parser.add_argument(
        "--mode", choices=["quick", "standard", "thorough"], default="standard",
        help="Benchmark mode: quick(1 shape), standard(LLM shapes), thorough(all shapes+dtypes)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--min-time", type=float, default=0.1,
        help="Minimum measurement time in seconds",
    )
    parser.add_argument(
        "--output", default=None,
        help="JSON output file path",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Only output JSON (suppress table)",
    )
    parser.add_argument(
        "--no-json", action="store_true",
        help="Only output table (suppress JSON)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available operators and exit",
    )
    return parser.parse_args()


def resolve_devices(args):
    """Resolve device list from CLI args. Returns (devices, display_names)."""
    display_names = {}
    if args.devices:
        resolved = []
        for d in args.devices:
            torch_dev = _PLATFORM_TO_TORCH_DEVICE.get(d, d)
            if torch_dev not in display_names:
                display_names[torch_dev] = d
            if torch_dev not in resolved:
                resolved.append(torch_dev)
        available = get_available_devices()
        result = [d for d in resolved if d in available]
    else:
        result = get_available_devices()
        for d in result:
            display_names[d] = d
    return result, display_names


def resolve_dtypes(args):
    """Resolve dtype list from CLI args."""
    if args.dtype:
        return [_DTYPE_MAP[d] for d in args.dtype]
    if args.mode == "thorough":
        return [torch.float32, torch.float16, torch.bfloat16]
    return [torch.float16, torch.bfloat16]


def list_operators():
    """Print available operators."""
    native = discover_native_operators()
    fallback = discover_fallback_operators()

    print("\nAvailable Operators:")
    print("=" * 60)

    print(f"\n  Native operators ({len(native)}):")
    for op in native:
        print(f"    - {op}")

    if fallback:
        print(f"\n  Torch fallback operators ({len(fallback)}):")
        # Group by category
        for cat, cfg in FALLBACK_OP_CONFIGS.items():
            cat_ops = [op for op in cfg["ops"] if op in fallback]
            if cat_ops:
                print(f"    {cat}: {', '.join(cat_ops)}")
        other_fallback = [op for op in fallback
                          if not any(op in cfg["ops"]
                                     for cfg in FALLBACK_OP_CONFIGS.values())]
        if other_fallback:
            print(f"    other: {', '.join(other_fallback[:20])}")
            if len(other_fallback) > 20:
                print(f"           ... and {len(other_fallback) - 20} more")
    else:
        print("\n  Torch fallback operators: (metadata not available)")

    # ntops operators
    available_ntops = []
    for name in _NTOPS_OPS:
        pascal = _pascal(name)
        cls = getattr(infini.ops, pascal, None)
        if cls is not None and hasattr(cls, "active_implementation_indices"):
            slots = set()
            for dev in get_available_devices():
                slots.update(cls.active_implementation_indices(dev))
            if slots:
                available_ntops.append(name)
    print(f"\n  ntops operators ({len(available_ntops)}/{len(_NTOPS_OPS)} available):")
    print(f"    {', '.join(available_ntops)}")
    missing = [op for op in _NTOPS_OPS if op not in available_ntops]
    if missing:
        print(f"    Missing: {', '.join(missing)}")

    print(f"\n  Devices: {', '.join(get_available_devices())}")
    print("=" * 60)
    print()


def main():
    args = parse_args()

    if args.list:
        list_operators()
        return

    # Resolve config
    devices, display_names = resolve_devices(args)
    dtypes = resolve_dtypes(args)

    config = BenchmarkConfig(
        mode=args.mode,
        warmup=args.warmup,
        min_time=args.min_time,
        devices=devices,
        device_display_names=display_names,
        dtypes=dtypes,
        ops=args.ops or [],
        category=args.category,
    )

    print(f"\nInfiniOps Benchmark | Mode: {config.mode} | "
          f"Devices: {config.devices} | Dtypes: {[str(d) for d in config.dtypes]}")

    results = []

    # Run native benchmarks
    if config.category in ("native", "all"):
        print(f"\nRunning native operator benchmarks...")
        native_results = run_native_benchmarks(config)
        results.extend(native_results)
        n_ok = sum(1 for r in native_results if r.status == "ok")
        n_skip = sum(1 for r in native_results if r.status == "skip")
        n_err = sum(1 for r in native_results if r.status == "error")
        print(f"  Native: {n_ok} ok, {n_skip} skipped, {n_err} errors")

    # Run fallback benchmarks
    if config.category in ("torch", "all"):
        print(f"\nRunning torch fallback benchmarks...")
        fallback_results = run_fallback_benchmarks(config)
        results.extend(fallback_results)
        n_ok = sum(1 for r in fallback_results if r.status == "ok")
        n_skip = sum(1 for r in fallback_results if r.status == "skip")
        n_err = sum(1 for r in fallback_results if r.status == "error")
        print(f"  Torch fallback: {n_ok} ok, {n_skip} skipped, {n_err} errors")

    # Run ntops benchmarks
    if config.category in ("ntops", "all"):
        print(f"\nRunning ntops operator benchmarks...")
        ntops_results = run_ntops_benchmarks(config)
        results.extend(ntops_results)

    # Output
    if not args.json_only:
        for device in config.devices:
            device_results = [r for r in results if r.device == device]
            _format_table(device_results, device, config.mode,
                          display_name=config.device_display_names.get(device, device))

    if not args.no_json and args.output:
        output_path = _format_json(results, config, args.output)
        print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
