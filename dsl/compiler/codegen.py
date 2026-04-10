"""C++ code generation for backend wrapper files."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dsl.decorators import InfiniOpDef, ManualOpDef

# Backend identifiers used in Device::Type enum.
CUDA_LIKE_BACKENDS = ("nvidia", "metax", "iluvatar", "moore")

# Maps backend name → Device::Type enum suffix (PascalCase).
BACKEND_ENUM = {
    "nvidia": "Nvidia",
    "metax": "Metax",
    "iluvatar": "Iluvatar",
    "moore": "Moore",
    "ascend": "Ascend",
    "cambricon": "Cambricon",
    "cpu": "Cpu",
}


def _pascal_case(snake: str) -> str:
    return "".join(w.capitalize() for w in snake.split("_"))


def _to_snake(pascal: str) -> str:
    """Convert PascalCase to snake_case."""
    import re

    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", pascal).lower()


def _include_guard(backend: str, op_snake: str, filename: str) -> str:
    """Build an include guard matching the project convention."""
    stem = pathlib.Path(filename).stem
    suffix = pathlib.Path(filename).suffix.lstrip(".")

    # Example: INFINI_OPS_NVIDIA_ADD_KERNEL_H_
    parts = ["INFINI_OPS", backend.upper(), op_snake.upper(), stem.upper()]
    parts.append(f"{suffix.upper()}_" if suffix else "H_")

    return "_".join(parts)


# ---- CUDA-like wrapper generation ----------------------------------------


def _resolve_cuda_template_info(
    op: ManualOpDef | InfiniOpDef,
) -> tuple[str, str] | None:
    """Derive the shared CUDA template class name and include path.

    Returns ``(CudaClassName, include_path)`` or ``None`` if the operator
    does not use a shared CUDA template.
    """
    from dsl.decorators import InfiniOpDef, ManualOpDef

    if isinstance(op, InfiniOpDef):
        op_snake = _to_snake(op.name)

        return f"Cuda{op.name}", f"cuda/{op_snake}/kernel.h"

    cuda_entry = op.backends.get("cuda")

    if cuda_entry is None:
        return None

    if isinstance(cuda_entry, dict):
        # Complex BLAS-style entry: {"include": ..., "class": ..., "blas": True}
        return cuda_entry.get("class"), cuda_entry.get("include")

    # Simple string: "cuda/add/kernel.h" → CudaAdd (convention: Cuda + OpName).
    return f"Cuda{op.name}", cuda_entry


def generate_cuda_wrapper(
    op: ManualOpDef | InfiniOpDef,
    backend: str,
    impl_index: int | None = None,
) -> str:
    """Generate a CUDA-like backend wrapper header.

    For operators backed by a shared ``Cuda*<Runtime<...>>`` template.
    """
    op_snake = _to_snake(op.name)
    enum_name = BACKEND_ENUM[backend]
    guard = _include_guard(backend, op_snake, "kernel.h")

    info = _resolve_cuda_template_info(op)

    if info is None:
        raise ValueError(
            f"Operator `{op.name}` has no `cuda` entry in backends; "
            f"cannot generate a CUDA-like wrapper for `{backend}`."
        )

    cuda_class, cuda_include = info

    # Build the template specialization.
    device_type = f"Device::Type::k{enum_name}"

    if impl_index is not None:
        device_type += f", {impl_index}"

    # Collect includes — no blank lines between them (matches existing style).
    lines: list[str] = ["#include <utility>", ""]

    if backend == "moore":
        lines.append("// clang-format off")
        lines.append('#include "moore/polyfills.cuh"')
        lines.append("// clang-format on")
        lines.append("")

    lines.append(f'#include "{cuda_include}"')
    lines.append(f'#include "{backend}/caster.cuh"')

    if backend == "moore":
        lines.append('#include "moore/polyfills.cuh"')

    lines.append(f'#include "{backend}/runtime_.h"')

    includes_str = "\n".join(lines)

    return "\n".join([
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        includes_str,
        "",
        "namespace infini::ops {",
        "",
        "template <>",
        f"class Operator<{op.name}, {device_type}>",
        f"    : public {cuda_class}<Runtime<Device::Type::k{enum_name}>> {{",
        " public:",
        f"  using {cuda_class}<Runtime<Device::Type::k{enum_name}>>::{cuda_class};",
        "};",
        "",
        "}  // namespace infini::ops",
        "",
        "#endif",
        "",
    ])


def generate_blas_wrapper(
    op: ManualOpDef,
    backend: str,
    blas_class: str,
    blas_include: str,
    impl_index: int | None = None,
) -> str:
    """Generate a BLAS-based backend wrapper (e.g. GEMM via cuBLAS)."""
    op_snake = _to_snake(op.name)
    enum_name = BACKEND_ENUM[backend]

    # Derive filename from the blas_include (e.g. "metax/blas.h" → mcblas).
    filename = f"{backend.lower()}blas.h"
    guard = _include_guard(backend, op_snake, filename)

    device_type = f"Device::Type::k{enum_name}"

    if impl_index is not None:
        device_type += f", {impl_index}"

    return (
        f"#ifndef {guard}\n"
        f"#define {guard}\n"
        f"\n"
        f'#include "{blas_include}"\n'
        f'#include "{backend}/blas.h"\n'
        f"\n"
        f"namespace infini::ops {{\n"
        f"\n"
        f"template <>\n"
        f"class Operator<{op.name}, {device_type}>\n"
        f"    : public {blas_class}<Blas<Device::Type::k{enum_name}>> {{\n"
        f" public:\n"
        f"  using {blas_class}<Blas<Device::Type::k{enum_name}>>::{blas_class};\n"
        f"}};\n"
        f"\n"
        f"}}  // namespace infini::ops\n"
        f"\n"
        f"#endif\n"
    )


# ---- High-level generation entry point -----------------------------------


def generate_wrappers_for_op(
    op: ManualOpDef | InfiniOpDef,
    devices: list[str],
    output_dir: pathlib.Path,
) -> list[pathlib.Path]:
    """Generate backend wrapper files for an operator.

    Works for both ``@manual_op`` and ``@infini_op`` operators.
    For ``@infini_op``, the shared CUDA template is the generated
    ``cuda/<op>/kernel.h`` file.

    Returns a list of generated file paths.
    """
    from dsl.decorators import InfiniOpDef, ManualOpDef

    op_snake = _to_snake(op.name)
    generated: list[pathlib.Path] = []

    # Build an effective backends dict.
    if isinstance(op, ManualOpDef):
        backends = op.backends
    else:
        # For @infini_op, the CUDA kernel is auto-generated.
        backends = dict(op.manual_backends)
        backends["cuda"] = f"cuda/{op_snake}/kernel.h"

    for backend in devices:

        if backend not in CUDA_LIKE_BACKENDS:
            continue

        if backend not in backends and "cuda" not in backends:
            continue

        # Check for an explicit backend entry (overrides shared CUDA path).
        explicit = backends.get(backend)

        if explicit is not None and isinstance(explicit, str):
            # Explicit hand-written file — do not generate a wrapper.
            continue

        # Generate from shared CUDA template.
        content = generate_cuda_wrapper(op, backend)
        out_path = output_dir / backend / op_snake / "kernel.h"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        generated.append(out_path)

    return generated
