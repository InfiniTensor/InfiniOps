"""Generate InfiniOps PyTorch wrappers from ATen `native_functions.yaml`.

For each op listed in `scripts/torch_ops.yaml`, this script finds the `.out`
variant in PyTorch's `native_functions.yaml` (fetched on demand from the
PyTorch GitHub release matching `_PYTORCH_VERSION`), parses its schema,
and emits:

  - `generated/base/<op>.h` — the InfiniOps base class
    `class <Op> : public Operator<<Op>>`, with a constructor and pure-virtual
    `operator()` mirroring the ATen schema.
  - `generated/torch/<op>/<op>.h` and `<op>.cc` — the PyTorch backend
    `Operator<<Op>, kDev, 8>` that calls `at::<op>_out(out, ...)`.
  - `generated/torch_ops_metadata.json` — the kind (`unary` / `binary` /
    `binary_alpha`) of every successfully-generated op, consumed by the
    parametrized test suite.

Slot 8 is the reserved convention for PyTorch backends; slots 0-7 are
left for native or vendor implementations.  (The slot must also be > 0
to side-step a partial-specialization-after-instantiation conflict with
the primary template `Operator<<Op>>` instantiated at index 0.)

The generated files are not committed; CMake regenerates them at configure
time when `WITH_TORCH=ON`.
"""

import argparse
import dataclasses
import json
import pathlib
import re
import sys
import urllib.request

import yaml

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
_OPS_YAML_PATH = _SCRIPTS_DIR / "torch_ops.yaml"
_BASE_DIR = _REPO_ROOT / "src" / "base"
_GENERATED_DIR = _REPO_ROOT / "generated"
_GENERATED_BASE_DIR = _GENERATED_DIR / "base"
_GENERATED_TORCH_DIR = _GENERATED_DIR / "torch"
_METADATA_PATH = _GENERATED_DIR / "torch_ops_metadata.json"

# Reserved slot for PyTorch backends.  Native and vendor implementations
# claim slots 0-7; PyTorch wrappers always live at 8.
_PYTORCH_SLOT = 8

# ATen uses symbolic names for some `int`/`float` defaults (e.g.
# `reduction=Mean`).  Map them to C++ identifiers usable in a call.
_ENUM_DEFAULTS = {
    "Mean": "at::Reduction::Mean",
    "Sum": "at::Reduction::Sum",
    "Contiguous": "at::MemoryFormat::Contiguous",
}

# PyTorch release tag whose `native_functions.yaml` defines the schemas
# we generate against.  Bump in lockstep with the minimum PyTorch version
# the generated wrappers should target.
_PYTORCH_VERSION = "v2.4.0"
_ATEN_YAML_URL = (
    f"https://raw.githubusercontent.com/pytorch/pytorch/{_PYTORCH_VERSION}"
    "/aten/src/ATen/native/native_functions.yaml"
)
_ATEN_YAML_CACHE = (
    _REPO_ROOT / "generated" / ".cache" / f"native_functions-{_PYTORCH_VERSION}.yaml"
)

# Order matches the device list in existing hand-written torch backends
# (see `src/torch/add/add.cc`).
_DEVICE_TYPES = (
    "kCpu",
    "kNvidia",
    "kCambricon",
    "kAscend",
    "kMetax",
    "kMoore",
    "kIluvatar",
    "kKunlun",
    "kHygon",
    "kQy",
)

# YAML scalar-type tokens → C++ types.  Reference types (e.g. `const Scalar&`)
# are not used so the generated signatures match the existing hand-written
# ones, which pass by value to keep pybind11 binding generation simple.
_SCALAR_TYPE_MAP = {
    # `at::Scalar` is implicitly constructible from `double`, so we expose
    # scalars as `double` in the base class to keep it torch-independent.
    "Scalar": "double",
    "int": "int64_t",
    "bool": "bool",
    "float": "double",
    # `SymInt` / `SymInt[]` exist for `torch.compile` internals; at runtime
    # they're just `int64`/IntArrayRef.
    "SymInt": "int64_t",
}

# Optional ATen types we hide from the user-facing API and pass as
# `at::nullopt` at the call site.  Covers the common "full default"
# case for most reductions and activations.  Tensor-typed optionals are
# hardcoded to `nullopt` too (e.g. `binary_cross_entropy.weight`); ops
# that *require* a non-null tensor would need a separate path.
_HARDCODE_NULLOPT_TYPES = frozenset(
    {
        "Scalar?",
        "int?",
        "bool?",
        "float?",
        "str?",
        "ScalarType?",
        "MemoryFormat?",
        "Layout?",
        "Device?",
        "Generator?",
        "Tensor?",
        "Tensor?[]",
        "int[]?",
        "int[1]?",
        "int[2]?",
        "int[3]?",
        "SymInt?",
        "SymInt[]?",
        "SymInt[1]?",
        "SymInt[2]?",
        "SymInt[3]?",
        "float[]?",
    }
)


@dataclasses.dataclass
class Param:
    name: str
    aten_type: str
    default: str | None
    keyword_only: bool

    @property
    def is_tensor(self) -> bool:
        # Real tensors only.  `Tensor?` is optional and falls through to
        # the hidden-param path (substituted with `at::nullopt`).
        return self.aten_type == "Tensor" or self.aten_type.startswith("Tensor(")

    @property
    def is_out(self) -> bool:
        # Mutable tensors carry `!` in their alias annotation, e.g. `Tensor(a!)`.
        return self.is_tensor and "!" in self.aten_type

    @property
    def is_hardcoded_nullopt(self) -> bool:
        """If `True`, the param is omitted from the user-facing API and
        passed as `at::nullopt` to ATen."""
        return self.aten_type in _HARDCODE_NULLOPT_TYPES

    @property
    def is_hidden(self) -> bool:
        """True if the param is omitted from the user-facing API.  Covers
        hardcoded-nullopt plus `bool`s and `int`/`float`s with a numeric
        default (typical for `keepdim`-style flags and `reduction`-style
        enums).  Also hides `int[]`/`int[1]` with a `[]` default (empty
        dim list means "all dims" for reductions like `amax`).  `Scalar`
        defaults are kept visible so ops like `sub(..., alpha=1)` expose
        `alpha` meaningfully."""
        if self.is_hardcoded_nullopt:
            return True
        if self.aten_type == "bool" and self.default in {"False", "True"}:
            return True
        if self.aten_type in {"int", "float", "SymInt"} and self.default is not None:
            return True
        if (
            self.aten_type.startswith("int[") or self.aten_type.startswith("SymInt[")
        ) and self.default is not None:
            return True
        if self.aten_type == "str" and self.default is not None:
            return True
        return False

    def hidden_value(self) -> str:
        """C++ literal substituted for a hidden param in the ATen call."""
        if self.is_hardcoded_nullopt:
            return "at::nullopt"
        if self.default == "True":
            return "true"
        if self.default == "False":
            return "false"
        if self.aten_type.startswith(("int[", "SymInt[")) and self.default is not None:
            # `int[N]=[a, b, c]` → `{a, b, c}`; `int[N]=0` (scalar default
            # for list type) → `{0, 0, ...}` replicated to size N.
            if self.default.startswith("["):
                return "{" + self.default[1:-1] + "}"
            size_match = re.search(r"\[(\d+)\]", self.aten_type)
            n = int(size_match.group(1)) if size_match else 1
            return "{" + ", ".join([self.default] * n) + "}"
        if self.aten_type == "str" and self.default is not None:
            # YAML strings already come quoted (e.g. `'none'`).
            return self.default
        if self.aten_type in {"int", "float", "SymInt"} and self.default is not None:
            # Translate known ATen enum defaults to their C++ identifiers.
            return _ENUM_DEFAULTS.get(self.default, self.default)
        raise AssertionError(
            f"param {self.name!r} of type {self.aten_type!r} with default "
            f"{self.default!r} is not hidden"
        )

    @property
    def cpp_type(self) -> str:
        if self.is_tensor:
            return "Tensor"
        if self.is_hidden:
            # Not exposed — the ATen call substitutes a hardcoded value
            # so the `cpp_type` is irrelevant.
            return "void"
        bare = self.aten_type.rstrip("?")
        # Required `int[N]` / `SymInt[N]` (no default) — pybind11 accepts
        # a Python list of ints into `std::vector<int64_t>`, which ATen
        # promotes to `IntArrayRef` implicitly.
        if bare.startswith(("int[", "SymInt[")) or bare in {"int[]", "SymInt[]"}:
            return "std::vector<int64_t>"
        try:
            return _SCALAR_TYPE_MAP[bare]
        except KeyError as exc:
            raise NotImplementedError(
                f"unsupported ATen type {self.aten_type!r} for param {self.name!r}"
            ) from exc


@dataclasses.dataclass
class Op:
    aten_name: str
    overload: str
    params: list[Param]

    @property
    def pascal_name(self) -> str:
        return _snake_to_pascal(self.infini_name)

    @property
    def infini_name(self) -> str:
        """InfiniOps op name.  Includes the overload to disambiguate
        between schemas of the same ATen op
        (e.g. `pow.Tensor_Tensor_out` → `pow_tensor_tensor`,
        `pow.Tensor_Scalar_out` → `pow_tensor_scalar`)."""
        suffix = self.overload.removesuffix("_out") if self.overload else ""
        if suffix and suffix != "out":
            return f"{self.aten_name}_{suffix.lower()}"
        return self.aten_name

    @property
    def tensor_params(self) -> list[Param]:
        return [p for p in self.params if p.is_tensor]

    @property
    def out_params(self) -> list[Param]:
        """Mutable tensor outputs.  Most ops have one (`Tensor(a!) out`);
        multi-output ops like `frexp` or `sort` have several
        (`Tensor(a!) values`, `Tensor(b!) indices`)."""
        return [p for p in self.params if p.is_out]

    @property
    def out_param(self) -> Param:
        """Single-output convenience.  Asserts there's exactly one."""
        outs = self.out_params
        assert len(outs) == 1, f"op {self.aten_name!r} has {len(outs)} out tensors"
        return outs[0]

    @property
    def visible_params(self) -> list[Param]:
        """Params the wrapper exposes to the user; hidden ones (hardcoded
        optional nullopt, default-`False`/`True` bools) are filtered."""
        return [p for p in self.params if not p.is_hidden]

    @property
    def is_testable(self) -> bool:
        """Cheap structural check: at least one out tensor.  Generators
        like `arange` / `linspace` produce a tensor from scalars only —
        those are still testable (the test runs the torch reference for
        shape discovery)."""
        return bool(self.out_params)


_FUNC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"(?:\.(?P<overload>\w+))?"
    r"\((?P<args>.*)\)\s*->\s*.+$"
)

_ARG_RE = re.compile(
    r"^(?P<type>\S+(?:\([^)]*\))?\??)"  # type with optional alias and `?`
    r"\s+(?P<name>\w+)"
    r"(?:\s*=\s*(?P<default>.+))?$"
)


def _parse_func(func_str: str) -> Op:
    m = _FUNC_RE.match(func_str)
    if not m:
        raise ValueError(f"could not parse func: {func_str!r}")
    return Op(
        aten_name=m.group("name"),
        overload=m.group("overload") or "",
        params=_parse_args(m.group("args")),
    )


def _parse_args(args_str: str) -> list[Param]:
    params: list[Param] = []
    keyword_only = False
    for token in _split_args(args_str):
        if token == "*":
            keyword_only = True
            continue
        params.append(_parse_one_arg(token, keyword_only))
    return params


def _split_args(args_str: str) -> list[str]:
    """Split on top-level commas, respecting `(...)` and `[...]`."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in args_str:
        if ch in "([":
            depth += 1
            current.append(ch)
        elif ch in ")]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            piece = "".join(current).strip()
            if piece:
                parts.append(piece)
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_one_arg(token: str, keyword_only: bool) -> Param:
    m = _ARG_RE.match(token)
    if not m:
        raise ValueError(f"could not parse arg: {token!r}")
    return Param(
        name=m.group("name"),
        aten_type=m.group("type"),
        default=m.group("default"),
        keyword_only=keyword_only,
    )


def _snake_to_pascal(s: str) -> str:
    return "".join(p.capitalize() for p in s.split("_"))


def _load_aten_yaml() -> str:
    """Return the contents of `native_functions.yaml`, fetching and caching
    the version pinned by `_PYTORCH_VERSION` on the first call."""
    if not _ATEN_YAML_CACHE.exists():
        _ATEN_YAML_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"fetching `native_functions.yaml` ({_PYTORCH_VERSION})...",
            file=sys.stderr,
        )
        with urllib.request.urlopen(_ATEN_YAML_URL) as response:
            _ATEN_YAML_CACHE.write_bytes(response.read())
    return _ATEN_YAML_CACHE.read_text()


def _find_out_entries(entries: list[dict], op_name: str) -> list[dict]:
    """Return all out-variant entries for `op_name`, with the bare
    `<name>.out(` form first and overload-suffixed variants
    (e.g. `pow.Tensor_Tensor_out(`) after.  Callers iterate in order
    and pick the first one parseable into a supported `kind`."""
    bare_prefix = f"{op_name}.out("
    overloaded = re.compile(rf"^{re.escape(op_name)}\.\w+_out\(")
    bare: list[dict] = []
    others: list[dict] = []
    for entry in entries:
        func = entry.get("func", "")
        if func.startswith(bare_prefix):
            bare.append(entry)
        elif overloaded.match(func):
            others.append(entry)
    return bare + others


def _format_signature(op: Op, *, include_defaults: bool = False) -> str:
    parts = []
    for param in op.visible_params:
        prefix = "" if param.is_out else "const "
        text = f"{prefix}{param.cpp_type} {param.name}"
        if include_defaults and param.default is not None:
            text += f" = {_translate_default(param)}"
        parts.append(text)
    return ", ".join(parts)


def _translate_default(param: Param) -> str:
    """Translate a YAML default literal to a C++ literal."""
    raw = param.default
    if raw == "True":
        return "true"
    if raw == "False":
        return "false"
    if raw == "None":
        return "{}"
    return raw  # numeric literals (`0`, `1`, `1.0`) pass through


def _generate_base_header(op: Op) -> str:
    init_pieces = []
    member_decls = []
    for param in op.tensor_params:
        init_pieces.append(f"        {param.name}_shape_{{{param.name}.shape()}}")
        init_pieces.append(f"        {param.name}_strides_{{{param.name}.strides()}}")
        init_pieces.append(f"        {param.name}_type_{{{param.name}.dtype()}}")
        member_decls.append(f"  Tensor::Shape {param.name}_shape_;")
        member_decls.append(f"  Tensor::Strides {param.name}_strides_;")
        member_decls.append(f"  DataType {param.name}_type_;")
    # All out tensors share a device; use the first one.
    init_pieces.append(
        f"        device_index_{{{op.out_params[0].name}.device().index()}}"
    )
    member_decls.append("  int device_index_{0};")

    init_list = ",\n".join(init_pieces).lstrip()

    return _BASE_TEMPLATE.format(
        name_uc=op.infini_name.upper(),
        pascal=op.pascal_name,
        ctor_signature=_format_signature(op),
        init_list=init_list,
        op_call_signature=_format_signature(op),
        member_decls="\n".join(member_decls),
    )


def _generate_torch_header(op: Op) -> str:
    return _TORCH_HEADER_TEMPLATE.format(
        name_uc=op.infini_name.upper(),
        name=op.infini_name,
        pascal=op.pascal_name,
        op_call_signature=_format_signature(op),
        slot=_PYTORCH_SLOT,
    )


def _generate_torch_source(op: Op) -> str:
    conversion_lines = []
    for param in op.tensor_params:
        data_expr = (
            f"{param.name}.data()"
            if param.is_out
            else f"const_cast<void*>({param.name}.data())"
        )
        conversion_lines.append(
            f"  auto at_{param.name} = ToAtenTensor<kDev>(\n"
            f"      {data_expr}, {param.name}_shape_, {param.name}_strides_,\n"
            f"      {param.name}_type_, device_index_);"
        )

    # ATen `_out` form puts all out tensors first, then non-out params
    # in YAML order.  Hardcoded-nullopt params become `at::nullopt`.
    arg_order = op.out_params + [p for p in op.params if not p.is_out]

    def _render_arg(p):
        if p.is_hidden:
            return p.hidden_value()
        if p.is_tensor:
            return f"at_{p.name}"
        return p.name

    aten_args = ", ".join(_render_arg(p) for p in arg_order)

    instantiations = "\n".join(
        f"template class Operator<{op.pascal_name}, "
        f"Device::Type::{dev}, {_PYTORCH_SLOT}>;"
        for dev in _DEVICE_TYPES
    )

    return _TORCH_SOURCE_TEMPLATE.format(
        name=op.infini_name,
        pascal=op.pascal_name,
        op_call_signature=_format_signature(op),
        tensor_conversions="\n".join(conversion_lines),
        # `at::<aten_name>_out` resolves the right kernel via C++ overload
        # resolution from the argument types we pass.
        aten_call=f"{op.aten_name}_out({aten_args})",
        slot=_PYTORCH_SLOT,
        instantiations=instantiations,
    )


_BASE_TEMPLATE = """\
// AUTO-GENERATED by `scripts/generate_torch_ops.py` — DO NOT EDIT.
#ifndef INFINI_OPS_BASE_{name_uc}_H_
#define INFINI_OPS_BASE_{name_uc}_H_

#include "operator.h"

namespace infini::ops {{

class {pascal} : public Operator<{pascal}> {{
 public:
  {pascal}({ctor_signature})
      : {init_list} {{}}

  virtual void operator()({op_call_signature}) const = 0;

 protected:
{member_decls}
}};

}}  // namespace infini::ops

#endif
"""


_TORCH_HEADER_TEMPLATE = """\
// AUTO-GENERATED by `scripts/generate_torch_ops.py` — DO NOT EDIT.
#ifndef INFINI_OPS_TORCH_{name_uc}_H_
#define INFINI_OPS_TORCH_{name_uc}_H_

#include "base/{name}.h"

namespace infini::ops {{

template <Device::Type kDev>
class Operator<{pascal}, kDev, {slot}> : public {pascal} {{
 public:
  using {pascal}::{pascal};

  void operator()({op_call_signature}) const override;
}};

}}  // namespace infini::ops

#endif
"""


_TORCH_SOURCE_TEMPLATE = """\
// AUTO-GENERATED by `scripts/generate_torch_ops.py` — DO NOT EDIT.
#include "torch/{name}/{name}.h"

#include "torch/tensor_.h"

namespace infini::ops {{

template <Device::Type kDev>
void Operator<{pascal}, kDev, {slot}>::operator()({op_call_signature}) const {{
{tensor_conversions}

  at::{aten_call};
}}

{instantiations}

}}  // namespace infini::ops
"""


def _emit(op: Op) -> None:
    base_path = _GENERATED_BASE_DIR / f"{op.infini_name}.h"
    torch_dir = _GENERATED_TORCH_DIR / op.infini_name
    torch_header_path = torch_dir / f"{op.infini_name}.h"
    torch_source_path = torch_dir / f"{op.infini_name}.cc"

    _GENERATED_BASE_DIR.mkdir(parents=True, exist_ok=True)
    torch_dir.mkdir(parents=True, exist_ok=True)

    base_path.write_text(_generate_base_header(op))
    torch_header_path.write_text(_generate_torch_header(op))
    torch_source_path.write_text(_generate_torch_source(op))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ops",
        nargs="*",
        help="Override the op allowlist. If omitted, reads `scripts/torch_ops.yaml`.",
    )
    args = parser.parse_args()

    op_names = args.ops or yaml.safe_load(_OPS_YAML_PATH.read_text())
    aten_entries = yaml.safe_load(_load_aten_yaml())

    skipped: list[tuple[str, str]] = []
    metadata: list[dict] = []

    for name in op_names:
        if (_BASE_DIR / f"{name}.h").exists():
            skipped.append((name, "hand-written `src/base/<op>.h` already exists"))
            continue

        candidates = _find_out_entries(aten_entries, name)
        if not candidates:
            skipped.append((name, f"no `.out` variant for `{name}` in YAML"))
            continue

        usable: list[Op] = []
        last_reason = ""
        for entry in candidates:
            try:
                op = _parse_func(entry["func"])
                for param in op.params:
                    _ = param.cpp_type  # eagerly raise on unsupported types
            except (NotImplementedError, ValueError) as exc:
                last_reason = str(exc)
                continue
            if not op.is_testable:
                last_reason = "no testable tensor input/output pair"
                continue
            usable.append(op)

        if not usable:
            skipped.append((name, last_reason or "no usable overload"))
            continue

        # Emit one InfiniOps wrapper per usable overload — `pow.Tensor_Tensor_out`
        # and `pow.Tensor_Scalar_out` become distinct classes
        # (`PowTensorTensor`, `PowTensorScalar`) so users get the right
        # behaviour by naming the variant they want.
        for op in usable:
            _emit(op)
            metadata.append(
                {
                    "name": op.infini_name,
                    "aten_name": op.aten_name,
                    "params": [
                        {
                            "name": p.name,
                            "type": p.aten_type,
                            "is_tensor": p.is_tensor,
                            "is_out": p.is_out,
                        }
                        for p in op.visible_params
                    ],
                }
            )

    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    _METADATA_PATH.write_text(json.dumps({"ops": metadata}, indent=2) + "\n")

    print(f"generated {len(metadata)} ops: {[m['name'] for m in metadata]}")
    for name, reason in skipped:
        print(f"  skipped {name!r}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
