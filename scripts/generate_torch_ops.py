"""Generate InfiniOps PyTorch wrappers from ATen `native_functions.yaml`.

For each op listed in `scripts/torch_ops.yaml`, this script finds the `.out`
variant in PyTorch's locally installed `native_functions.yaml`, parses its schema,
and emits:

  - `generated/base/<op>.h` — the InfiniOps base class
    `class <Op> : public Operator<<Op>>`, with constructors and pure-virtual
    `operator()` overloads mirroring the selected ATen schemas.
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
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

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

# Default PyTorch schema label used only in diagnostics when CMake does not
# provide the locally installed torch version.  Codegen reads the actual schema
# from installed `torchgen/packaged/ATen/native/native_functions.yaml`.
_DEFAULT_PYTORCH_VERSION = "v2.4.0"

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
    # `str` for required string params (e.g. `index_reduce.reduce`).
    # `std::string` marshals through pybind11 cleanly and converts
    # implicitly to ATen's `c10::string_view`.
    "str": "std::string",
}

# `Dimname` overloads (named-tensor dim) are skipped — passing them
# from Python to ATen requires a wrapper conversion through
# `at::Dimname::fromSymbol(...)` that doesn't fit the cleanly-rendered
# 1:1 arg model, and named tensors remain experimental in PyTorch.
# The int-dim overload is always emitted alongside, so we lose nothing
# user-visible.

# Optional ATen types we hide from the user-facing API and pass as a
# typed empty optional at the call site.  Covers the common "full
# default" case for most reductions and activations.  We use a typed
# `c10::optional<T>{}` rather than bare `at::nullopt` so the compiler
# can disambiguate ops with multiple `_out` overloads (e.g. `clamp_out`
# accepts both `optional<Scalar>` and `optional<Tensor>` for `min`/`max`).
_NULLOPT_BY_TYPE = {
    "Scalar?": "c10::optional<at::Scalar>{}",
    "int?": "c10::optional<int64_t>{}",
    "bool?": "c10::optional<bool>{}",
    "float?": "c10::optional<double>{}",
    "str?": "c10::optional<c10::string_view>{}",
    "ScalarType?": "c10::optional<at::ScalarType>{}",
    "MemoryFormat?": "c10::optional<at::MemoryFormat>{}",
    "Layout?": "c10::optional<at::Layout>{}",
    "Device?": "c10::optional<at::Device>{}",
    "Generator?": "c10::optional<at::Generator>{}",
    "Tensor?": "c10::optional<at::Tensor>{}",
    "Tensor?[]": "c10::List<c10::optional<at::Tensor>>{}",
    "int[]?": "c10::optional<at::IntArrayRef>{}",
    "int[1]?": "c10::optional<at::IntArrayRef>{}",
    "int[2]?": "c10::optional<at::IntArrayRef>{}",
    "int[3]?": "c10::optional<at::IntArrayRef>{}",
    "SymInt?": "c10::optional<int64_t>{}",
    "SymInt[]?": "c10::optional<at::IntArrayRef>{}",
    "SymInt[1]?": "c10::optional<at::IntArrayRef>{}",
    "SymInt[2]?": "c10::optional<at::IntArrayRef>{}",
    "SymInt[3]?": "c10::optional<at::IntArrayRef>{}",
    "float[]?": "c10::optional<at::ArrayRef<double>>{}",
}
_HARDCODE_NULLOPT_TYPES = frozenset(_NULLOPT_BY_TYPE)


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
    def is_mutable_tensor(self) -> bool:
        # Mutable tensors carry `!` in their alias annotation, e.g.
        # `Tensor(a!)`.

        return self.is_tensor and "!" in self.aten_type

    @property
    def is_out(self) -> bool:
        # In ATen `_out` schemas, output tensors are keyword-only mutable tensor
        # params.  Some mutable tensors are real inputs (`running_mean` /
        # `running_var` in `_batch_norm_with_update`) and must stay in schema
        # order, so mutability alone is not enough.

        return self.is_mutable_tensor and self.keyword_only

    @property
    def is_hardcoded_nullopt(self) -> bool:
        """If `True`, the param is omitted from the user-facing API and
        passed as `at::nullopt` to ATen."""

        return self.aten_type in _HARDCODE_NULLOPT_TYPES

    @property
    def is_hidden(self) -> bool:
        """True if the param is omitted from the user-facing API.

        Default-valued non-optional params (\\`bool\\`, \\`int\\`, \\`float\\`,
        \\`str\\`, \\`int[N]\\`, …) used to be hidden as a convenience, but
        reviewers consistently flagged the resulting omissions —
        \\`bool upper/transpose/unitriangular\\` on \\`triangular_solve\\`,
        \\`int diagonal\\` on \\`triu\\`, \\`str ord\\` on \\`linalg_matrix_norm\\`,
        \\`int n\\` on the special chebyshev family, etc. — as missing
        semantic controls.  They are now exposed and forwarded to ATen.

        Optional ATen types (\\`Tensor?\\`, \\`Scalar?\\`, \\`int?\\`, …) remain
        hidden for now — exposing them would require teaching the torch
        source to thread \\`std::optional\\` through to ATen, which is a
        separate refactor.  The same goes for ATen-internal types like
        \\`Generator?\\`/\\`Layout?\\` that have no InfiniOps analogue.
        """

        return self.is_hardcoded_nullopt

    def hidden_value(self) -> str:
        """C++ literal substituted for a hidden param in the ATen call."""

        if self.is_hardcoded_nullopt:
            return _NULLOPT_BY_TYPE[self.aten_type]

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
            # YAML uses single-quoted strings (e.g. `'none'`); C++ char
            # literals also use single quotes, so swap to doubles.

            return '"' + self.default.strip("'\"") + '"'

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
            # `Tensor[]` / `Tensor(a!)[]` would need `std::vector<Tensor>` and a
            # different ATen call shape — not yet supported, so reject so the
            # whole overload gets skipped instead of emitting code that calls
            # `at::<op>_out(at::Tensor, ...)` against an `at::TensorList`
            # signature.
            if self.aten_type.endswith("[]"):
                raise NotImplementedError(
                    f"`Tensor[]` param {self.name!r} not supported yet"
                )

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
        """InfiniOps public op name.

        ATen disambiguates `_out` overloads with suffixes like `Tensor_Tensor_out`,
        `out_x`, `forward_output`, `grad_input`, but reviewers consistently
        flag those suffixes as bad public-API naming when they leak into
        InfiniOps class names.  Different ATen overloads of the same base op
        become overloaded `operator()` methods on a single class instead.  When
        two overloads collapse to the same visible C++ signature after hidden
        defaults, `_dedupe_visible_overloads` keeps only one.

        ATen-internal leading underscores and in-place trailing underscores are
        also normalized so generated class names do not collide with existing
        public ops (`_softmax` → `AtenSoftmax`, `add_` → `AddInplace`).
        """
        return _public_op_name(self.aten_name)

    @property
    def is_inplace(self) -> bool:
        """True for ATen's single-underscore in-place variants."""

        return _is_inplace_aten_name(self.aten_name)

    @property
    def tensor_params(self) -> list[Param]:
        return [p for p in self.params if p.is_tensor]

    @property
    def out_params(self) -> list[Param]:
        """Mutable tensor outputs.  Most ops have one (`Tensor(a!) out`);
        multi-output ops like `frexp` or `sort` have several
        (`Tensor(a!) values`, `Tensor(b!) indices`)."""

        if self.is_inplace:
            return [self.params[0]]

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
        """Cheap structural check: at least one out tensor, and the first
        constructor parameter is a tensor.  The latter is needed because
        `Operator::Make(Tensor tensor, Args... args)` dispatches on
        `tensor.device()`, so an op like `pow.Scalar_out(Scalar self,
        Tensor exponent, *, Tensor(a!) out)` cannot be wired up without
        a separate dispatch path.  Generators like `arange` / `linspace`
        also fall under this rule (no input tensors at all)."""

        if not self.out_params:
            return False

        if self.is_inplace:
            return self.params[0].is_mutable_tensor

        # `params` includes out tensors at the end; check the first
        # non-out param.  If there are no non-out params (`empty.out`,
        # `arange.out`), this op also fails the dispatch precondition.
        non_out = [p for p in self.params if not p.is_out]

        if not non_out:
            return False

        return non_out[0].is_tensor


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

    name = m.group("name")
    # ATen names the first tensor parameter `self` (matching the
    # method-style \`tensor.abs()\` convention).  InfiniOps uses
    # \`input\` for the primary tensor input across all hand-written
    # bases (\`Add\`, \`Gemm\`, …) per \`CONTRIBUTING.md\` §C++.
    # Rename at parse time so the generated headers match.
    if name == "self":
        name = "input"

    return Param(
        name=name,
        aten_type=m.group("type"),
        default=m.group("default"),
        keyword_only=keyword_only,
    )


def _snake_to_pascal(s: str) -> str:
    return "".join(p.capitalize() for p in s.split("_"))


def _is_inplace_aten_name(name: str) -> bool:
    """Return whether `name` is an ATen in-place operator name."""

    return name.endswith("_") and not name.endswith("__")


def _public_op_name(aten_name: str) -> str:
    """Map ATen-only spelling to stable InfiniOps public names."""

    public_name = aten_name
    prefix = ""

    if public_name.startswith("_"):
        prefix = "aten_"
        public_name = public_name.lstrip("_")

    if _is_inplace_aten_name(public_name):
        public_name = public_name[:-1] + "_inplace"

    return prefix + public_name


def _base_path(op_name: str) -> pathlib.Path:
    return _BASE_DIR / f"{op_name}.h"


def _load_aten_yaml(version: str) -> str:
    """Return the `native_functions.yaml` bundled with installed `torchgen`.

    `WITH_TORCH=ON` already requires a local PyTorch installation. PyTorch
    wheels ship the matching ATen schema under `torchgen/packaged`, including
    vendor forks, so codegen should not depend on fetching PyTorch sources from
    GitHub during CI builds.
    """

    packaged = _load_packaged_aten_yaml()

    if packaged is None:
        raise RuntimeError(
            "could not find installed `torchgen` packaged "
            f"`native_functions.yaml` for PyTorch schema {version!r}"
        )

    print(
        "using installed `torchgen` packaged `native_functions.yaml` "
        f"for PyTorch schema {version}.",
        file=sys.stderr,
    )

    return packaged


def _load_packaged_aten_yaml() -> str | None:
    """Return the `native_functions.yaml` bundled with installed `torchgen`.

    PyTorch wheels install `torchgen/packaged/ATen/native/native_functions.yaml`;
    using it lets offline CI images generate wrappers against the exact schema
    shipped with their installed torch fork.
    """

    spec = importlib.util.find_spec("torchgen")

    if spec is None or spec.submodule_search_locations is None:
        return None

    for root in spec.submodule_search_locations:
        path = (
            pathlib.Path(root)
            / "packaged"
            / "ATen"
            / "native"
            / "native_functions.yaml"
        )

        if path.is_file():
            return path.read_text()

    return None


def _find_out_entries(entries: list[dict], op_name: str) -> list[dict]:
    """Return all out-variant entries for `op_name`, with the bare
    `<name>.out(` form first and overload-suffixed variants
    (e.g. `pow.Tensor_Tensor_out(`, `kthvalue.values(`) after.  An
    entry counts as an out-variant when it (a) is named
    `<op_name>.out`, (b) ends in `_out`, or (c) carries a
    `Tensor(<letter>!)` mutability annotation — that last case covers
    multi-output ops named after their output tensors
    (`kthvalue.values`, `mode.values`, …)."""
    bare_prefix = f"{op_name}.out("
    op_overload = re.compile(rf"^{re.escape(op_name)}\.\w+\(")
    mut_tensor = re.compile(r"Tensor\([a-z]!\)")
    bare: list[dict] = []
    others: list[dict] = []

    for entry in entries:
        func = entry.get("func", "")

        if func.startswith(bare_prefix):
            bare.append(entry)
        elif op_overload.match(func) and (
            func.split("(", 1)[0].endswith("_out") or mut_tensor.search(func)
        ):
            others.append(entry)

    return bare + others


def _format_signature(op: Op, *, include_defaults: bool = False) -> str:
    parts = []

    for param in op.visible_params:
        prefix = "" if param.is_mutable_tensor else "const "
        text = f"{prefix}{param.cpp_type} {param.name}"

        if include_defaults and param.default is not None:
            text += f" = {_translate_default(param)}"

        parts.append(text)

    return ", ".join(parts)


def _visible_signature_key(op: Op) -> tuple[str, ...]:
    """C++ overload identity for the user-facing API.

    Parameter names and top-level `const` do not distinguish C++ overloads, so
    only the exposed C++ type sequence participates in duplicate detection.
    """

    return tuple(param.cpp_type for param in op.visible_params)


def _canonical_overload_score(index: int, op: Op) -> tuple[bool, int, int, str, int]:
    """Sort key for duplicate visible signatures.

    Prefer the canonical unsuffixed InfiniOps name, then the schema that hides
    fewer ATen-only defaults, then the shorter deterministic name.
    """

    return (
        op.infini_name != op.aten_name,
        sum(param.is_hidden for param in op.params),
        len(op.infini_name),
        op.infini_name,
        index,
    )


def _dedupe_visible_overloads(ops: list[Op]) -> tuple[list[Op], list[tuple[Op, Op]]]:
    """Drop overloads that collapse to the same visible C++ signature.

    Returns the selected overloads in the original schema order plus a list of
    `(skipped, kept)` duplicate pairs for diagnostics.
    """
    winners: dict[tuple[str, ...], tuple[int, Op]] = {}
    duplicates: list[tuple[Op, tuple[str, ...]]] = []

    for index, op in enumerate(ops):
        key = _visible_signature_key(op)
        current = winners.get(key)

        if current is None:
            winners[key] = (index, op)
            continue

        current_index, current_op = current

        if _canonical_overload_score(index, op) < _canonical_overload_score(
            current_index, current_op
        ):
            duplicates.append((current_op, key))
            winners[key] = (index, op)
        else:
            duplicates.append((op, key))

    selected_indices = {index for index, _ in winners.values()}
    selected = [op for index, op in enumerate(ops) if index in selected_indices]
    duplicate_pairs = [
        (skipped, winners[key][1])
        for skipped, key in duplicates
        if winners[key][1] is not skipped
    ]

    return selected, duplicate_pairs


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


def _generate_base_header(name: str, ops: list[Op]) -> str:
    pascal = _snake_to_pascal(name)

    member_decls = []
    tensor_member_order = []
    seen_tensor_members = set()
    scalar_member_order = []
    scalar_member_types = {}

    for op in ops:
        for param in op.tensor_params:
            if param.name in seen_tensor_members:
                continue

            seen_tensor_members.add(param.name)
            tensor_member_order.append(param.name)
            member_decls.append(f"  Tensor::Shape {param.name}_shape_;")
            member_decls.append(f"  Tensor::Strides {param.name}_strides_;")
            member_decls.append(f"  DataType {param.name}_type_;")

        # Visible non-tensor params (scalars, strings, vectors) are also
        # stored on the base so backends can dispatch on them later — not
        # only at the moment `operator()` is invoked.  Reviewers flagged
        # this on multiple PRs (e.g. `n` on
        # `special_chebyshev_polynomial_v_n_scalar`).  Same-named params
        # across overloads must share a type; if they conflict, the second
        # overload's member is dropped (later constructors leave it
        # default-initialised).
        for param in op.visible_params:
            if param.is_tensor or param.name in scalar_member_types:
                continue

            scalar_member_order.append(param.name)
            scalar_member_types[param.name] = param.cpp_type
            member_decls.append(f"  {param.cpp_type} {param.name}_{{}};")

    member_decls.append("  int device_index_{0};")

    constructors = []
    calls = []

    for op in ops:
        init_pieces = []
        tensor_params = {param.name: param for param in op.tensor_params}
        scalar_params = {
            param.name: param
            for param in op.visible_params
            if not param.is_tensor
            and scalar_member_types.get(param.name) == param.cpp_type
        }

        for param_name in tensor_member_order:
            param = tensor_params.get(param_name)

            if param is None:
                continue

            init_pieces.append(f"        {param.name}_shape_{{{param.name}.shape()}}")
            init_pieces.append(
                f"        {param.name}_strides_{{{param.name}.strides()}}"
            )
            init_pieces.append(f"        {param.name}_type_{{{param.name}.dtype()}}")

        for param_name in scalar_member_order:
            param = scalar_params.get(param_name)

            if param is None:
                continue

            init_pieces.append(f"        {param.name}_{{{param.name}}}")

        # All out tensors share a device; use the first one.  Keep this last
        # so initializer order follows the member declaration order.
        init_pieces.append(
            f"        device_index_{{{op.out_params[0].name}.device().index()}}"
        )

        init_list = ",\n".join(init_pieces).lstrip()
        constructors.append(
            f"  {pascal}({_format_signature(op)})\n      : {init_list} {{}}"
        )
        calls.append(f"  virtual void operator()({_format_signature(op)}) const = 0;")

    return _BASE_TEMPLATE.format(
        name_uc=name.upper(),
        pascal=pascal,
        constructors="\n\n".join(constructors),
        op_calls="\n\n".join(calls),
        member_decls="\n\n".join(member_decls),
    )


def _generate_torch_header(name: str, ops: list[Op]) -> str:
    pascal = _snake_to_pascal(name)
    op_calls = "\n\n".join(
        f"  void operator()({_format_signature(op)}) const override;" for op in ops
    )

    return _TORCH_HEADER_TEMPLATE.format(
        name_uc=name.upper(),
        name=name,
        pascal=pascal,
        op_calls=op_calls,
        slot=_PYTORCH_SLOT,
    )


def _generate_torch_method_source(name: str, op: Op) -> str:
    pascal = _snake_to_pascal(name)
    conversion_lines = []

    for param in op.tensor_params:
        data_expr = (
            f"{param.name}.data()"
            if param.is_mutable_tensor
            else f"const_cast<void*>({param.name}.data())"
        )
        conversion_lines.append(
            f"  auto at_{param.name} = ToAtenTensor<kDev>(\n"
            f"      {data_expr}, {param.name}_shape_, {param.name}_strides_,\n"
            f"      {param.name}_type_, device_index_);"
        )

    def _render_arg(p):
        if p.is_hidden:
            return p.hidden_value()

        if p.is_tensor:
            return f"at_{p.name}"

        return p.name

    if op.is_inplace:
        # In-place ATen calls keep the mutable input in positional order,
        # unlike `_out` calls which place output tensors first.
        input_param = op.params[0]
        arg_order = op.params[1:]
        aten_call = (
            f"at_{input_param.name}.{op.aten_name}"
            f"({', '.join(_render_arg(p) for p in arg_order)})"
        )
    else:
        # ATen `_out` form puts all out tensors first, then non-out params
        # in YAML order.  Hardcoded-nullopt params become `at::nullopt`.
        arg_order = op.out_params + [p for p in op.params if not p.is_out]
        aten_call = (
            f"at::{op.aten_name}_out({', '.join(_render_arg(p) for p in arg_order)})"
        )

    return _TORCH_METHOD_TEMPLATE.format(
        pascal=pascal,
        op_call_signature=_format_signature(op),
        tensor_conversions="\n".join(conversion_lines),
        # The generated call expression resolves the right kernel via C++
        # overload resolution from the argument types we pass.
        aten_call=aten_call,
        slot=_PYTORCH_SLOT,
    )


def _generate_torch_source(name: str, ops: list[Op]) -> str:
    pascal = _snake_to_pascal(name)
    methods = "\n\n".join(_generate_torch_method_source(name, op) for op in ops)
    # Guard each explicit instantiation by the matching `WITH_<DEV>` macro
    # so a build that only enables a subset of devices does not pay the
    # ATen template-instantiation cost (and memory pressure) for the
    # devices it does not link against.  Each macro is set by
    # `target_compile_definitions` in `src/CMakeLists.txt`.
    instantiations = "\n".join(
        f"#ifdef WITH_{dev.removeprefix('k').upper()}\n"
        f"template class Operator<{pascal}, Device::Type::{dev}, {_PYTORCH_SLOT}>;\n"
        f"#endif"
        for dev in _DEVICE_TYPES
    )

    return _TORCH_SOURCE_TEMPLATE.format(
        name=name,
        methods=methods,
        instantiations=instantiations,
    )


_BASE_TEMPLATE = """\
#ifndef INFINI_OPS_BASE_{name_uc}_H_
#define INFINI_OPS_BASE_{name_uc}_H_

#include "operator.h"

namespace infini::ops {{

class {pascal} : public Operator<{pascal}> {{
 public:
{constructors}

{op_calls}

 protected:
{member_decls}
}};

}}  // namespace infini::ops

#endif
"""


_TORCH_HEADER_TEMPLATE = """\
#ifndef INFINI_OPS_TORCH_{name_uc}_H_
#define INFINI_OPS_TORCH_{name_uc}_H_

#include "base/{name}.h"

namespace infini::ops {{

template <Device::Type kDev>
class Operator<{pascal}, kDev, {slot}> : public {pascal} {{
 public:
  using {pascal}::{pascal};

{op_calls}
}};

}}  // namespace infini::ops

#endif
"""


_TORCH_METHOD_TEMPLATE = """\
template <Device::Type kDev>
void Operator<{pascal}, kDev, {slot}>::operator()({op_call_signature}) const {{
{tensor_conversions}

  {aten_call};
}}
"""


_TORCH_SOURCE_TEMPLATE = """\
#include "torch/{name}/{name}.h"

#include "torch/tensor_.h"

namespace infini::ops {{

{methods}

{instantiations}

}}  // namespace infini::ops
"""


def _find_clang_format() -> str | None:
    """Return the path to a `clang-format` binary, or `None` if unavailable.

    Generated files live under `generated/`, which is gitignored, so offline CI
    containers should not block or try network access just to format them.
    """

    found = shutil.which("clang-format")

    if found:
        return found

    print(
        "`clang-format` not found on PATH; generated files will be emitted "
        "without formatting.",
        file=sys.stderr,
    )

    return None


def _clang_format(text: str, path: pathlib.Path) -> str:
    """Pipe `text` through `clang-format` so generated headers / sources
    satisfy the same style check (`clang-format` v21) that CI runs.
    `path` informs include sorting (the file's own header should come
    first in a `.cc`).  If no `clang-format` binary is available, return
    the input unchanged."""

    if _CLANG_FORMAT is None:
        return text

    return subprocess.run(
        [_CLANG_FORMAT, f"--assume-filename={path}"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _emit(name: str, ops: list[Op], *, emit_base: bool) -> None:
    base_path = _GENERATED_BASE_DIR / f"{name}.h"
    torch_dir = _GENERATED_TORCH_DIR / name
    torch_header_path = torch_dir / f"{name}.h"
    torch_source_path = torch_dir / f"{name}.cc"

    if emit_base:
        _GENERATED_BASE_DIR.mkdir(parents=True, exist_ok=True)
        base_path.write_text(_clang_format(_generate_base_header(name, ops), base_path))

    torch_dir.mkdir(parents=True, exist_ok=True)

    torch_header_path.write_text(
        _clang_format(_generate_torch_header(name, ops), torch_header_path)
    )
    torch_source_path.write_text(
        _clang_format(_generate_torch_source(name, ops), torch_source_path)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ops",
        nargs="*",
        help="Override the op allowlist. If omitted, reads `scripts/torch_ops.yaml`.",
    )
    parser.add_argument(
        "--pytorch-version",
        default=os.environ.get("INFINIOPS_PYTORCH_VERSION", _DEFAULT_PYTORCH_VERSION),
        help=(
            "PyTorch version label used in diagnostics. Schemas are read from "
            "the locally installed `torchgen` package. Default: `%(default)s`. "
            "Can also be set via the `INFINIOPS_PYTORCH_VERSION` environment "
            "variable."
        ),
    )
    args = parser.parse_args()

    global _CLANG_FORMAT
    _CLANG_FORMAT = _find_clang_format()

    op_names = args.ops or yaml.safe_load(_OPS_YAML_PATH.read_text())
    aten_entries = yaml.safe_load(_load_aten_yaml(args.pytorch_version))

    # Wipe previous outputs so files for ops that have since been removed,
    # renamed, or rejected by `cpp_type` don't linger and get picked up by
    # the CMake glob.  Both `generated/base/` and `generated/torch/` are
    # written exclusively by this script.
    if _GENERATED_BASE_DIR.exists():
        shutil.rmtree(_GENERATED_BASE_DIR)

    if _GENERATED_TORCH_DIR.exists():
        shutil.rmtree(_GENERATED_TORCH_DIR)

    skipped: list[tuple[str, str]] = []
    metadata: list[dict] = []

    for name in op_names:
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

        usable, duplicate_overloads = _dedupe_visible_overloads(usable)

        for skipped_op, kept_op in duplicate_overloads:
            skipped.append(
                (
                    skipped_op.infini_name,
                    "duplicate visible C++ signature for "
                    f"`{name}`; using `{kept_op.infini_name}`",
                )
            )

        # Emit one InfiniOps wrapper per ATen op.  Distinct visible overloads
        # become overloaded constructors / `operator()` methods on the same
        # class (`Pow` exposes both tensor and scalar exponents).  Overloads
        # that collapse to the same C++ signature after hidden defaults are
        # skipped above.  When a hand-written `src/base/<name>.h` exists,
        # skip emitting `generated/base/<name>.h` so the hand-written one
        # wins (the generated torch source's `#include "base/<name>.h"`
        # resolves through `src/` first).  Signature mismatches surface as
        # compile errors with a clear message — drop the op from the YAML
        # to suppress.
        public_name = usable[0].infini_name
        _emit(public_name, usable, emit_base=not _base_path(public_name).exists())

        for op in usable:
            metadata.append(
                {
                    "name": public_name,
                    "aten_name": op.aten_name,
                    "overload_name": op.infini_name,
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

    generated_names = sorted({m["name"] for m in metadata})
    print(
        f"generated {len(metadata)} overloads across {len(generated_names)} ops: "
        f"{generated_names}"
    )

    for name, reason in skipped:
        print(f"  skipped {name!r}: {reason}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
