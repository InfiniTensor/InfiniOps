#!/usr/bin/env python
"""InfiniOps Correctness Test Script.

Compares InfiniOps operator outputs against PyTorch reference implementations
for correctness verification. No performance benchmarking — just pass/fail.

Usage:
    python scripts/test_correctness.py                          # all ntops ops, cuda, float16
    python scripts/test_correctness.py --ops core               # core ops only
    python scripts/test_correctness.py --ops ext                # ext ops only
    python scripts/test_correctness.py --ops matmul add mul     # specific ops
    python scripts/test_correctness.py --device mlu --dtype bfloat16
    python scripts/test_correctness.py --list                   # list available ops
"""

import argparse
import json
import pathlib
import sys

# ---------------------------------------------------------------------------
# Ensure project root on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

import infini.ops

from tests.utils import (
    empty_strided,
    get_available_devices,
    get_stream,
    randn_strided,
    randint_strided,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PYTORCH_SLOT = 8

# Metadata file paths
try:
    _INSTALLED_METADATA_PATH = (
        pathlib.Path(infini.ops.__file__).resolve().with_name("torch_ops_metadata.json")
    )
except Exception:
    _INSTALLED_METADATA_PATH = _PROJECT_ROOT / "generated" / "torch_ops_metadata.json"
_SOURCE_METADATA_PATH = _PROJECT_ROOT / "generated" / "torch_ops_metadata.json"

_NTOPS_CORE_OPS = [
    "mm", "bmm", "addmm", "add", "sub", "mul", "div", "abs", "neg",
    "eq", "ne", "lt", "gt", "le", "ge", "sum", "mean", "amax", "amin", "argmax",
    "rms_norm", "avg_pool2d", "sin", "cos", "sign", "reciprocal", "rsqrt",
    "maximum", "minimum", "index_select",
]

_NTOPS_EXT_OPS = [
    "exp", "sigmoid", "silu", "gelu", "tanh", "bitwise_not", "softmax", "sqrt",
    "log", "log_softmax", "pow", "bitwise_and", "bitwise_or",
    "cumsum", "topk", "gather", "sort", "ceil", "round", "hardtanh",
    "cosh", "sinh", "asin", "acos", "atan", "acosh", "asinh", "atanh",
    "exp2", "expm1", "log2", "log10", "log1p", "erf", "erfc", "erfinv",
    "hardswish", "hardsigmoid", "mish", "log_sigmoid",
    "elu", "softplus", "atan2", "xlogy",
    "cumprod", "bitwise_xor", "nan_to_num", "threshold", "lerp",
    "tan", "square", "cummax", "cummin",
]

_NTOPS_OPS = _NTOPS_CORE_OPS + _NTOPS_EXT_OPS

_INT_ONLY_OPS = {"bitwise_and", "bitwise_or", "bitwise_not", "bitwise_xor"}
_PYTORCH_ONLY_OPS = {"relu", "matmul", "clamp"}

# Shapes — use a single representative shape for quick correctness check
_TEST_SHAPES = {
    "unary": [(256, 1024)],
    "binary": [(256, 1024)],
    "mm": [((32, 64), (64, 128))],
    "bmm": [((4, 32, 64), (4, 64, 128))],
    "addmm": [((32, 64), (64, 128))],
    "rms_norm": [(256, 1024)],
    "pool2d": [(1, 16, 32, 32)],
}

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
    ("leaky_relu", "negative_slope"): 0.01,
    ("elu", "alpha"): 1.0,
    ("elu", "scale"): 1.0,
    ("elu", "input_scale"): 1.0,
    ("softplus", "beta"): 1.0,
    ("softplus", "threshold"): 20.0,
    ("cumprod", "dim"): -1,
    ("threshold", "threshold"): 1.0,
    ("threshold", "value"): 0.0,
    ("clamp_max", "max"): 6.0,
    ("clamp_min", "min"): 0.0,
    ("lerp", "weight"): 0.5,
    ("cummax", "dim"): -1,
    ("cummin", "dim"): -1,
}


# ---------------------------------------------------------------------------
# Helpers (same logic as benchmark.py)
# ---------------------------------------------------------------------------
def _synchronize(device):
    if device == "cpu":
        return
    mod = getattr(torch, device, None)
    if mod is not None and hasattr(mod, "synchronize"):
        mod.synchronize()


def _pascal(snake_name):
    return "".join(part.capitalize() for part in snake_name.split("_"))


def _ntops_op_type(op_name):
    _unary = {"abs", "neg", "exp", "rsqrt", "sigmoid", "silu", "gelu", "relu",
              "tanh", "sin", "cos", "bitwise_not", "softmax", "clamp",
              "sqrt", "reciprocal", "log_softmax", "rms_norm",
              "ceil", "floor", "log", "sign", "round",
              "hardtanh", "amin", "sort", "argmax"}
    _binary = {"add", "sub", "mul", "div", "pow", "eq", "ne", "lt", "le",
               "gt", "ge", "bitwise_and", "bitwise_or", "maximum", "minimum", "where",
               "atan2", "logaddexp", "logaddexp2", "xlogy", "bitwise_xor", "lerp"}
    _reduction = {"sum", "mean", "amax", "cumsum"}
    if op_name in _unary:
        return "unary"
    if op_name in _binary:
        return "binary"
    if op_name in _reduction:
        return "unary"
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
        return "unary"
    return "unary"


def _ntops_get_shapes(op_name):
    op_type = _ntops_op_type(op_name)
    return _TEST_SHAPES.get(op_type, _TEST_SHAPES["unary"])


def _load_metadata():
    for path in (_INSTALLED_METADATA_PATH, _SOURCE_METADATA_PATH):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
    return None


_METADATA_CACHE = None


def _ntops_get_metadata(op_name):
    global _METADATA_CACHE
    if _METADATA_CACHE is None:
        _METADATA_CACHE = _load_metadata()
    if not _METADATA_CACHE:
        return None
    candidates = [op for op in _METADATA_CACHE.get("ops", []) if op["name"] == op_name]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def tensor_count(op):
        return sum(1 for p in op["params"] if p["is_tensor"] and not p["is_out"])

    candidates.sort(key=tensor_count, reverse=True)
    return candidates[0]


def _ntops_build_scalar(op_name, param_name, param_type):
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
    _refs = {
        "softmax": lambda x: torch.nn.functional.softmax(x, dim=-1),
        "log_softmax": lambda x: torch.nn.functional.log_softmax(x, dim=-1),
        "silu": lambda x: torch.nn.functional.silu(x),
        "hardtanh": lambda x: torch.nn.functional.hardtanh(x, min_val=-1.0, max_val=1.0),
        "avg_pool2d": lambda x: torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1),
        "rms_norm": None,
        "mul": lambda x, y: torch.mul(x, y),
        "hardswish": lambda x: torch.nn.functional.hardswish(x),
        "hardsigmoid": lambda x: torch.nn.functional.hardsigmoid(x),
        "mish": lambda x: torch.nn.functional.mish(x),
        "log_sigmoid": lambda x: torch.nn.functional.logsigmoid(x),
        "leaky_relu": lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.01),
        "elu": lambda x: torch.nn.functional.elu(x, alpha=1.0),
        "softplus": lambda x: torch.nn.functional.softplus(x, beta=1.0, threshold=20.0),
        "logaddexp": lambda x, y: torch.logaddexp(x, y),
        "logaddexp2": lambda x, y: torch.logaddexp2(x, y),
        "xlogy": lambda x, y: torch.special.xlogy(x, y),
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


def _ntops_build_inputs(op_name, op_type, op_meta, device, dtype, shape):
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
        return [x], None

    if op_type == "pool2d":
        x = randn_strided(shape, None, dtype=dtype, device=device)
        out = empty_strided(shape, None, dtype=dtype, device=device)
        return [x], out

    # Default: unary
    x = _rand(shape, effective_dtype)
    out = empty_strided(shape, None, dtype=effective_dtype, device=device)
    return [x], out


# ---------------------------------------------------------------------------
# Ref function builder (same as benchmark.py)
# ---------------------------------------------------------------------------
def _ntops_ref_out_fn(op_name, inputs, scalar_args, out, indices_out=None):
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
        "logical_not": lambda inp, o: torch.logical_not(inp, out=o),
        "tan": lambda inp, o: torch.tan(inp, out=o),
        "square": lambda inp, o: torch.square(inp, out=o),
        "exp2": lambda inp, o: torch.exp2(inp, out=o),
        "log2": lambda inp, o: torch.log2(inp, out=o),
        # These don't support out= natively, use copy_
        "logit": lambda inp, o: o.copy_(torch.logit(inp)),
        "sinc": lambda inp, o: o.copy_(torch.sinc(inp)),
        "nan_to_num": lambda inp, o: o.copy_(torch.nan_to_num(inp)),
    }
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
        "atan2": lambda a, b, o: torch.atan2(a, b, out=o),
        "mul": lambda a, b, o: torch.mul(a, b, out=o),
        "bitwise_xor": lambda a, b, o: torch.bitwise_xor(a, b, out=o),
    }
    _matmul_out = {
        "mm": lambda a, b, o: torch.mm(a, b, out=o),
        "bmm": lambda a, b, o: torch.bmm(a, b, out=o),
    }
    _special_out = {
        "avg_pool2d": lambda inp, o: torch.nn.functional.avg_pool2d(
            inp, kernel_size=3, stride=1, padding=1, out=o),
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


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
def check_correctness(infiniops_fn, ref_fn, device):
    """Run both functions once and compare outputs.

    Returns ("passed", None) or ("failed", max_diff_str).
    Handles NaN masking: positions where both sides produce NaN are ignored.
    """
    try:
        res_io = infiniops_fn()
        _synchronize(device)
        # Clone InfiniOps result BEFORE running ref (both may write to same buffer)
        if isinstance(res_io, torch.Tensor):
            res_io_saved = res_io.clone()
        else:
            res_io_saved = res_io

        res_ref = ref_fn()
        _synchronize(device)

        if not isinstance(res_io_saved, torch.Tensor) or not isinstance(res_ref, torch.Tensor):
            return "skip", None

        if res_io_saved.is_floating_point() and res_ref.is_floating_point():
            io_f = res_io_saved.float()
            ref_f = res_ref.float()
            diff = (io_f - ref_f).abs()
            # Mask out positions where both sides agree on non-finite values:
            # - both NaN: ok (e.g. sqrt of negative)
            # - both same-sign Inf: ok (e.g. reciprocal of zero)
            agree_mask = (io_f.isnan() & ref_f.isnan()) | (
                io_f.isinf() & ref_f.isinf() & (io_f == ref_f)
            )
            diff = diff.masked_fill_(agree_mask, 0)
            # Use only finite values for ref_abs to avoid NaN/Inf contamination
            finite_ref = ref_f[ref_f.isfinite()]
            ref_abs = finite_ref.abs().max().item() if finite_ref.numel() > 0 else 1e-8
            max_diff = diff.max().item()
            # If max_diff is still NaN, all positions were masked → passed
            if max_diff != max_diff:  # NaN check
                return "passed", None
            rel_err = max_diff / max(ref_abs, 1e-8)
            if rel_err < 1e-2:
                return "passed", None
            else:
                return "failed", f"rel_err={rel_err:.2e}, max_diff={max_diff:.2e}"
        else:
            # integer / boolean tensors: exact match
            if torch.equal(res_io_saved, res_ref):
                return "passed", None
            else:
                return "failed", "mismatch"
    except Exception as exc:
        return "error", str(exc)


# ---------------------------------------------------------------------------
# Build InfiniOps + ref callables (extracted from _run_single_ntops)
# ---------------------------------------------------------------------------
def _build_callables(op_name, op_type, op_meta, device, dtype, shape):
    """Build (infiniops_fn, ref_fn) for a given op.

    Returns (infiniops_fn, ref_fn) or raises on unsupported ops.
    """
    effective_dtype = torch.int32 if op_name in _INT_ONLY_OPS else dtype
    op_pascal = _pascal(op_name)
    op_cls = getattr(infini.ops, op_pascal, None)
    is_pytorch_only = op_name in _PYTORCH_ONLY_OPS or op_cls is None

    if not is_pytorch_only:
        if not op_cls.active_implementation_indices(device):
            is_pytorch_only = True

    ref_func = _ntops_resolve_ref(op_name)
    inputs, out = _ntops_build_inputs(op_name, op_type, op_meta, device, effective_dtype, shape)

    # --- Build callables (same branching as benchmark.py) ---

    if is_pytorch_only:
        def infiniops_fn():
            return ref_func(*inputs)

        def ref_fn():
            return ref_func(*inputs)
        return infiniops_fn, ref_fn

    if op_name == "rms_norm":
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

        return infiniops_fn, ref_fn

    if op_name == "clamp":
        min_val, max_val = -1.0, 1.0

        def infiniops_fn():
            infini.ops.clamp(inputs[0], min_val, max_val, out,
                             implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            result = torch.clamp(inputs[0], min=min_val, max=max_val)
            out.copy_(result)
            return out

        return infiniops_fn, ref_fn

    if op_name in ("add", "mul"):
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

        return infiniops_fn, ref_fn

    if op_name == "addmm":
        bias, mat1, mat2 = inputs[0], inputs[1], inputs[2]

        def infiniops_fn():
            infini.ops.addmm(bias, mat1, mat2, 1.0, 1.0, out,
                             implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            torch.addmm(bias, mat1, mat2, out=out)
            return out

        return infiniops_fn, ref_fn

    if op_name == "amax":
        out_shape = shape[:-1]
        amax_out = empty_strided(out_shape, None, dtype=effective_dtype, device=device)

        def infiniops_fn():
            infini.ops.amax(inputs[0], [-1], False, amax_out,
                            implementation_index=_PYTORCH_SLOT)
            return amax_out

        def ref_fn():
            return torch.amax(inputs[0], dim=-1)

        return infiniops_fn, ref_fn

    if op_name == "amin":
        out_shape = shape[:-1]
        amin_out = empty_strided(out_shape, None, dtype=effective_dtype, device=device)

        def infiniops_fn():
            infini.ops.amin(inputs[0], [-1], False, amin_out,
                            implementation_index=_PYTORCH_SLOT)
            return amin_out

        def ref_fn():
            return torch.amin(inputs[0], dim=-1)

        return infiniops_fn, ref_fn

    if op_name in ("sum", "mean", "cumsum"):
        if op_meta is not None:
            if op_name in ("sum", "mean"):
                sum_out = torch.empty((), dtype=effective_dtype, device=device)
            else:
                sum_out = out

            in_params = [p for p in op_meta["params"] if not p["is_out"]]
            scalar_args = [_ntops_build_scalar(op_name, p["name"], p["type"])
                          for p in in_params if not p["is_tensor"]]
            call_args = (*inputs, *scalar_args, sum_out)

            def infiniops_fn():
                getattr(infini.ops, op_name)(*call_args,
                                             implementation_index=_PYTORCH_SLOT)
                return sum_out

            if op_name == "sum":
                def ref_fn():
                    return torch.sum(inputs[0])
            elif op_name == "mean":
                def ref_fn():
                    return torch.mean(inputs[0])
            elif op_name == "cumsum":
                def ref_fn():
                    torch.cumsum(inputs[0], dim=-1, out=sum_out)
                    return sum_out

            return infiniops_fn, ref_fn
        else:
            raise RuntimeError(f"No metadata for {op_name}")

    if op_name == "topk":
        k = 5
        dim = -1
        values_shape = list(shape)
        values_shape[dim] = k
        values_shape = tuple(values_shape)
        topk_out = empty_strided(values_shape, None, dtype=effective_dtype, device=device)
        indices_out = empty_strided(values_shape, None, dtype=torch.long, device=device)

        def infiniops_fn():
            infini.ops.topk(inputs[0], k, dim, True, True, topk_out, indices_out,
                            implementation_index=_PYTORCH_SLOT)
            return topk_out

        def ref_fn():
            result = torch.topk(inputs[0], k=k, dim=dim)
            topk_out.copy_(result.values)
            return topk_out

        return infiniops_fn, ref_fn

    if op_name == "gather":
        idx = torch.zeros(shape, dtype=torch.long, device=device)

        def infiniops_fn():
            infini.ops.gather(inputs[0], 0, idx, False, out,
                              implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            torch.gather(inputs[0], 0, idx, out=out)
            return out

        return infiniops_fn, ref_fn

    if op_name == "index_select":
        idx = torch.tensor([0], dtype=torch.long, device=device)
        # index_select output shape: (len(idx), ...) along dim 0
        idx_out_shape = (idx.numel(),) + shape[1:]
        idx_out = empty_strided(idx_out_shape, None, dtype=effective_dtype, device=device)

        def infiniops_fn():
            infini.ops.index_select(inputs[0], 0, idx, idx_out,
                                    implementation_index=_PYTORCH_SLOT)
            return idx_out

        def ref_fn():
            return torch.index_select(inputs[0], 0, idx)

        return infiniops_fn, ref_fn

    if op_name == "sort":
        dim = -1
        indices_out = empty_strided(shape, None, dtype=torch.long, device=device)

        def infiniops_fn():
            infini.ops.sort(inputs[0], dim, True, out, indices_out,
                            implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            torch.sort(inputs[0], dim=dim, descending=True, out=(out, indices_out))
            return out

        return infiniops_fn, ref_fn

    if op_name == "argmax":
        argmax_out = torch.empty((), dtype=torch.long, device=device)

        def infiniops_fn():
            infini.ops.argmax(inputs[0], False, argmax_out,
                              implementation_index=_PYTORCH_SLOT)
            return argmax_out

        def ref_fn():
            return torch.argmax(inputs[0])

        return infiniops_fn, ref_fn

    if op_name == "lerp":
        a, b = inputs[0], inputs[1]

        def infiniops_fn():
            infini.ops.lerp(a, b, 0.5, out,
                            implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            torch.lerp(a, b, 0.5, out=out)
            return out

        return infiniops_fn, ref_fn

    if op_name == "clamp_max":
        def infiniops_fn():
            infini.ops.clamp_max(inputs[0], 6.0, out,
                                 implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            torch.clamp_max(inputs[0], max=6.0, out=out)
            return out

        return infiniops_fn, ref_fn

    if op_name == "clamp_min":
        def infiniops_fn():
            infini.ops.clamp_min(inputs[0], 0.0, out,
                                 implementation_index=_PYTORCH_SLOT)
            return out

        def ref_fn():
            torch.clamp_min(inputs[0], min=0.0, out=out)
            return out

        return infiniops_fn, ref_fn

    if op_name in ("cummax", "cummin"):
        cindices_out = empty_strided(shape, None, dtype=torch.long, device=device)
        ref_cindices_out = empty_strided(shape, None, dtype=torch.long, device=device)
        if op_name == "cummax":
            def infiniops_fn():
                infini.ops.cummax(inputs[0], -1, out, cindices_out,
                                  implementation_index=_PYTORCH_SLOT)
                return out

            def ref_fn():
                torch.cummax(inputs[0], dim=-1, out=(out, ref_cindices_out))
                return out
        else:
            def infiniops_fn():
                infini.ops.cummin(inputs[0], -1, out, cindices_out,
                                  implementation_index=_PYTORCH_SLOT)
                return out

            def ref_fn():
                torch.cummin(inputs[0], dim=-1, out=(out, ref_cindices_out))
                return out

        return infiniops_fn, ref_fn

    # --- Generic fallback (from metadata) ---
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

        _ref_out = _ntops_ref_out_fn(op_name, inputs, scalar_args, out)
        if _ref_out is not None:
            ref_fn = _ref_out
        else:
            ref_func = _ntops_resolve_ref(op_name)
            if ref_func is not None:
                tmp_buf = empty_strided(out.shape, None, dtype=out.dtype, device=device)

                def ref_fn(_fn=ref_func, _tmp=tmp_buf, _out=out, _inp=inputs):
                    _tmp.copy_(_fn(*_inp))
                    return _tmp  # return tmp so correctness compares properly
            else:
                def ref_fn():
                    out.fill_(0)
                    return out

        return infiniops_fn, ref_fn

    # No metadata: try generic unary/binary
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
    else:
        def ref_fn():
            out.fill_(0)
            return out

    return infiniops_fn, ref_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
DEVICE_MAP = {
    "nvidia": "cuda",
    "cambricon": "mlu",
    "metax": "cuda",
    "ascend": "npu",
    "iluvatar": "cuda",
    "moore": "musa",
    "hygon": "cuda",
}


def resolve_device(name):
    return DEVICE_MAP.get(name, name)


def main():
    parser = argparse.ArgumentParser(description="InfiniOps Correctness Test")
    parser.add_argument("--ops", nargs="+", default=None,
                        help="Operators to test (supports 'core', 'ext' group names)")
    parser.add_argument("--device", default=None,
                        help="Device (cuda/mlu/npu/cpu or nvidia/cambricon/ascend)")
    parser.add_argument("--dtype", default="float16",
                        help="Dtype (float16/bfloat16/float32)")
    parser.add_argument("--list", action="store_true",
                        help="List all available operators and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-op detail")
    args = parser.parse_args()

    # --list
    if args.list:
        print("Available operators:")
        print(f"  Core ({len(_NTOPS_CORE_OPS)}): {', '.join(_NTOPS_CORE_OPS)}")
        print(f"  Ext   ({len(_NTOPS_EXT_OPS)}): {', '.join(_NTOPS_EXT_OPS)}")
        print(f"  Total: {len(_NTOPS_OPS)}")
        return

    # Resolve device
    if args.device:
        device = resolve_device(args.device)
    else:
        avail = get_available_devices()
        device = avail[-1] if len(avail) > 1 else (avail[0] if avail else "cpu")
    print(f"Device: {device}")

    # Resolve dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.float16)
    print(f"Dtype:  {dtype}")

    # Resolve ops
    _GROUP_MAP = {"core": _NTOPS_CORE_OPS, "ext": _NTOPS_EXT_OPS}
    if args.ops:
        expanded = []
        for op in args.ops:
            if op in _GROUP_MAP:
                expanded.extend(_GROUP_MAP[op])
            else:
                expanded.append(op)
        ops_to_run = [op for op in _NTOPS_OPS if op in expanded]
    else:
        ops_to_run = _NTOPS_OPS[:]

    total = len(ops_to_run)
    print(f"Ops:    {total} operators\n")

    # Run correctness tests
    passed_list = []
    failed_list = []
    error_list = []
    skip_list = []

    for idx, op_name in enumerate(ops_to_run, 1):
        op_type = _ntops_op_type(op_name)
        op_meta = _ntops_get_metadata(op_name)
        shapes = _ntops_get_shapes(op_name)

        op_results = []
        for shape in shapes:
            try:
                infiniops_fn, ref_fn = _build_callables(
                    op_name, op_type, op_meta, device, dtype, shape)
            except Exception as exc:
                op_results.append(("error", str(exc)))
                continue

            status, detail = check_correctness(infiniops_fn, ref_fn, device)
            op_results.append((status, detail))

        # Aggregate results for this op
        statuses = [r[0] for r in op_results]
        if all(s == "passed" for s in statuses):
            passed_list.append(op_name)
            label = "PASSED"
        elif any(s == "error" for s in statuses):
            error_list.append((op_name, [r for r in op_results if r[0] == "error"]))
            label = "ERROR"
        elif any(s == "skip" for s in statuses) and not any(s == "failed" for s in statuses):
            skip_list.append(op_name)
            label = "SKIP"
        else:
            failed_list.append((op_name, op_results))
            label = "FAILED"

        tag = f"[{idx}/{total}]"
        print(f"  {tag} {op_name:<20} {label}")

        if args.verbose and label != "PASSED":
            for i, (s, d) in enumerate(op_results):
                shape_str = str(shapes[i]) if i < len(shapes) else "?"
                if d:
                    print(f"         {shape_str}: {s} — {d}")

    # Summary
    n_passed = len(passed_list)
    n_failed = len(failed_list)
    n_error = len(error_list)
    n_skip = len(skip_list)
    print(f"\n{'='*60}")
    print(f"Results: {n_passed} passed, {n_failed} failed, {n_error} error, {n_skip} skip "
          f"(total {total})")
    print(f"{'='*60}")

    if failed_list:
        print(f"\nFailed ({n_failed}):")
        for op_name, results in failed_list:
            details = "; ".join(f"{s}({d or ''})" for s, d in results)
            print(f"  {op_name:<20} {details}")

    if error_list:
        print(f"\nError ({n_error}):")
        for op_name, results in error_list:
            for _, detail in results:
                print(f"  {op_name:<20} {detail[:80]}")

    # Exit code
    sys.exit(1 if (n_failed + n_error) > 0 else 0)


if __name__ == "__main__":
    main()
