"""Unified test for every operator emitted by `generate_torch_ops.py`.

The generator writes `generated/torch_ops_metadata.json` listing every op
with full per-parameter info (`name`, `type`, `is_tensor`, `is_out`).
A single parametrized test reads that metadata, builds inputs from the
parameter list, calls the InfiniOps wrapper and the torch reference, and
compares each output tensor.  Adding an op to `scripts/torch_ops.yaml`
extends coverage with no test changes.
"""

import json
import pathlib

import infini.ops
import pytest
import torch

from tests.utils import randn_strided

# PyTorch backends are emitted at this slot — see `_PYTORCH_SLOT` in
# `scripts/generate_torch_ops.py`.
_PYTORCH_SLOT = 8

_METADATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "generated"
    / "torch_ops_metadata.json"
)
_METADATA = (
    json.loads(_METADATA_PATH.read_text()) if _METADATA_PATH.exists() else {"ops": []}
)

_SHAPES = (
    (13, 4),
    (13, 4, 4),
    (4, 4, 5632),
)

_DTYPES = (
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-2, 1e-2),
    (torch.bfloat16, 1e-2, 1e-2),
)

# Op-specific input shapes for matrix ops (`mm` etc.) which cannot use
# `randn_strided(shape)` for both inputs.  The tuple is one shape per
# tensor input, in YAML order.
_TENSOR_SHAPES = {
    "mm": ((8, 16), (16, 12)),
    "bmm": ((4, 8, 16), (4, 16, 12)),
    "matmul": ((8, 16), (16, 12)),
    "dot": ((16,), (16,)),
    "vdot": ((16,), (16,)),
    "mv": ((8, 16), (16,)),
    "inner": ((8, 16), (8, 16)),
    "outer": ((8,), (12,)),
    "ger": ((8,), (12,)),
    "kron": ((3, 4), (2, 3)),
}

# Per-(op, param-name) values for non-tensor inputs.  Lookup falls back
# to a type-based default if no entry exists.
_SCALAR_VALUES = {
    ("clamp_min", "min"): -0.5,
    ("clamp_max", "max"): 0.5,
    ("leaky_relu", "negative_slope"): 0.01,
    ("hardshrink", "lambd"): 0.5,
    ("softshrink", "lambd"): 0.5,
    ("mvlgamma", "p"): 2,
    ("prod", "dim"): 0,
    ("cumsum", "dim"): 0,
    ("cumprod", "dim"): 0,
    ("logcumsumexp", "dim"): 0,
    ("cummax", "dim"): 0,
    ("cummin", "dim"): 0,
    ("softmax", "dim"): -1,
    ("log_softmax", "dim"): -1,
    ("threshold", "threshold"): 0.0,
    ("threshold", "value"): 0.0,
    ("hardtanh", "min_val"): -1.0,
    ("hardtanh", "max_val"): 1.0,
    ("softplus", "beta"): 1.0,
    ("softplus", "threshold"): 20.0,
    ("elu", "alpha"): 1.0,
    ("elu", "scale"): 1.0,
    ("elu", "input_scale"): 1.0,
    ("sub", "alpha"): 1.0,
    ("addcmul", "value"): 1.0,
    ("addcdiv", "value"): 1.0,
}

_TYPE_DEFAULTS = {"int": 0, "bool": False}

# Errors emitted by upstream PyTorch and vendor-forked variants for
# unsupported (op, dtype, device) combinations.  We skip rather than fail
# on these — the gap is in PyTorch, not InfiniOps.
_VENDOR_SKIP_PATTERNS = (
    "not implemented for",  # upstream PyTorch
    "CNNL_STATUS_BAD_PARAM",  # `torch_mlu` (Cambricon)
    "MUDNN failed",  # `torch_musa` (Moore)
    "Could not run",  # missing dispatcher entry on this backend
    "don't support tensor dtype",  # `torch_mlu` dtype check
    "result requires dtype",  # output dtype mismatch (e.g. `float_power`)
)

# Full reductions with low-precision inputs diverge between the functional
# (`torch.<op>(x)`) and `_out` paths because of intermediate-precision
# choices we cannot align from outside ATen.
_LARGE_REDUCTION_OPS = frozenset(
    {"sum", "mean", "nansum", "nanmean", "prod", "std", "var"}
)


def _torch_func(op_name):
    """Resolve the reference function across `torch`, `torch.special`,
    and `torch.nn.functional`.  `special_<x>` falls through to
    `torch.special.<x>` with the prefix stripped."""
    candidates = [
        (torch, op_name),
        (torch.special, op_name),
        (torch.nn.functional, op_name),
    ]
    if op_name.startswith("special_"):
        candidates.append((torch.special, op_name.removeprefix("special_")))
    for namespace, attr in candidates:
        func = getattr(namespace, attr, None)
        if func is not None:
            return func
    pytest.skip(f"no reference function for `{op_name}` in PyTorch")


def _pascal(snake_name):
    return "".join(part.capitalize() for part in snake_name.split("_"))


def _skip_if_not_active(op_name, device):
    op_class = getattr(infini.ops, _pascal(op_name), None)
    if op_class is None:
        pytest.skip(f"`{op_name}` class not exposed on this build")
    if _PYTORCH_SLOT not in op_class.active_implementation_indices(device):
        pytest.skip(f"`{op_name}` slot {_PYTORCH_SLOT} not active on `{device}`")


def _skip_low_precision_reduction(op_name, dtype, device):
    if op_name in _LARGE_REDUCTION_OPS:
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip(f"`{op_name}` precision diverges on fp16/bf16")
        if device == "musa":
            pytest.skip(f"`{op_name}` on `torch_musa` diverges from CPU reference")


def _build_input_value(op_name, param, shape, dtype, device, tensor_idx):
    """Build the value passed to a non-out parameter."""
    if param["is_tensor"]:
        per_op = _TENSOR_SHAPES.get(op_name)
        tshape = per_op[tensor_idx] if per_op is not None else shape
        return randn_strided(tshape, None, dtype=dtype, device=device)
    key = (op_name, param["name"])
    if key in _SCALAR_VALUES:
        return _SCALAR_VALUES[key]
    return _TYPE_DEFAULTS.get(param["type"], 0.5)


def _call_infini(op_name, *args):
    try:
        getattr(infini.ops, op_name)(*args, implementation_index=_PYTORCH_SLOT)
    except RuntimeError as exc:
        if any(p in str(exc) for p in _VENDOR_SKIP_PATTERNS):
            pytest.skip(f"`{op_name}` unsupported by torch on this device/dtype")
        raise


def _assert_close(actual, expected, rtol, atol):
    if actual.dtype.is_floating_point:
        assert torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    else:
        assert torch.equal(actual, expected)


def _testable_ops():
    """Filter out ops the harness can't drive — currently just bool-output
    ops, since InfiniOps `DataType` has no `kBool`.  Unknown until runtime,
    so we skip-at-test-time rather than filter here."""
    return _METADATA.get("ops", [])


@pytest.mark.parametrize("op_meta", _testable_ops(), ids=lambda m: m["name"])
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPES)
def test_op(op_meta, shape, dtype, device, rtol, atol):
    op_name = op_meta["name"]
    _skip_if_not_active(op_name, device)
    _skip_low_precision_reduction(op_name, dtype, device)

    in_params = [p for p in op_meta["params"] if not p["is_out"]]
    out_params = [p for p in op_meta["params"] if p["is_out"]]

    # Build inputs in YAML order.
    inputs = []
    tensor_idx = 0
    for p in in_params:
        inputs.append(_build_input_value(op_name, p, shape, dtype, device, tensor_idx))
        if p["is_tensor"]:
            tensor_idx += 1

    # Run the reference to discover output shape(s)/dtype(s).
    try:
        ref = _torch_func(op_name)(*inputs)
    except (RuntimeError, TypeError) as exc:
        pytest.skip(f"`torch.{op_name}` rejects these inputs: {exc}")

    ref_outs = ref if isinstance(ref, tuple) else (ref,)
    assert len(ref_outs) == len(out_params), (
        f"`{op_name}` produced {len(ref_outs)} outputs but the schema declares "
        f"{len(out_params)}"
    )

    if any(t.dtype == torch.bool for t in ref_outs):
        pytest.skip(f"`{op_name}` returns `bool` — InfiniOps `DataType` has no `kBool`")

    outs = [torch.empty_like(t) for t in ref_outs]
    _call_infini(op_name, *inputs, *outs)

    for actual, expected in zip(outs, ref_outs):
        _assert_close(actual, expected, rtol, atol)
