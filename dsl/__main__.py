"""CLI entry point: ``python -m dsl``."""

from __future__ import annotations

import argparse
import difflib
import pathlib
import sys

from dsl.compiler.codegen import CUDA_LIKE_BACKENDS, generate_wrappers_for_op
from dsl.compiler.infini_codegen import generate_cpu_kernel, generate_cuda_kernel
from dsl.compiler.parser import parse_infini_op
from dsl.compiler.patterns import match_dag
from dsl.compiler.registry import REGISTRY
from dsl.decorators import InfiniOpDef
from dsl.ops import discover


def _to_snake(pascal: str) -> str:
    """Convert PascalCase to snake_case."""
    import re

    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", pascal).lower()


def _generate_infini_op(
    op: InfiniOpDef,
    output_dir: pathlib.Path,
) -> list[pathlib.Path]:
    """Generate CUDA + CPU files for an `@infini_op` operator."""
    dag = parse_infini_op(op)
    match = match_dag(dag)
    op_snake = _to_snake(op.name)
    generated: list[pathlib.Path] = []

    # Generate shared CUDA kernel.
    cuda_content = generate_cuda_kernel(op, dag, match)
    cuda_path = output_dir / "cuda" / op_snake / "kernel.h"
    cuda_path.parent.mkdir(parents=True, exist_ok=True)
    cuda_path.write_text(cuda_content)
    generated.append(cuda_path)

    # Generate CPU implementation.
    cpu_content = generate_cpu_kernel(op, dag, match)
    cpu_path = output_dir / "cpu" / op_snake / f"{op_snake}.h"
    cpu_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_path.write_text(cpu_content)
    generated.append(cpu_path)

    return generated


def _diff_file(expected: str, actual: str, label: str) -> list[str]:
    return list(
        difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=f"existing/{label}",
            tofile=f"generated/{label}",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="InfiniOps DSL compiler — generate backend wrappers.",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=list(CUDA_LIKE_BACKENDS),
        help="CUDA-like backends to generate wrappers for.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("generated"),
        help="Output directory for generated files.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare generated wrappers against existing hand-written files "
        "in src/ and report differences.",
    )
    parser.add_argument(
        "--ops",
        nargs="*",
        default=None,
        help="Generate only the specified operators (default: all).",
    )

    args = parser.parse_args()

    # Discover and register all operator definitions.
    discover()

    ops = REGISTRY.all_ops()

    if args.ops:
        ops = {k: v for k, v in ops.items() if k in args.ops}

    if not ops:
        print("No operators found.", file=sys.stderr)
        sys.exit(1)

    src_dir = pathlib.Path("src")
    total_generated = 0
    total_diffs = 0

    for name, op in sorted(ops.items()):

        if isinstance(op, InfiniOpDef):
            generated = _generate_infini_op(op, args.output)
            # Also generate CUDA-like backend wrappers for @infini_op.
            generated += generate_wrappers_for_op(op, args.devices, args.output)
        else:
            generated = generate_wrappers_for_op(op, args.devices, args.output)

        total_generated += len(generated)

        if args.verify:

            for gen_path in generated:
                # Map generated path to the existing hand-written path in src/.
                rel = gen_path.relative_to(args.output)
                existing_path = src_dir / rel

                if not existing_path.exists():
                    print(f"NEW  {rel}")
                    total_diffs += 1

                    continue

                expected = gen_path.read_text()
                actual = existing_path.read_text()

                if expected != actual:
                    diff = _diff_file(expected, actual, str(rel))
                    print(f"DIFF {rel}")

                    for line in diff:
                        print(line, end="")

                    print()
                    total_diffs += 1
                else:
                    print(f"OK   {rel}")

    if args.verify:
        print(f"\n{total_generated} files checked, {total_diffs} differences.")

        if total_diffs:
            sys.exit(1)
    else:
        print(f"Generated {total_generated} wrapper files in {args.output}/")


if __name__ == "__main__":
    main()
