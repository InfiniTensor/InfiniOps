"""CLI entry point: ``python -m dsl``."""

from __future__ import annotations

import argparse
import difflib
import pathlib
import sys

from dsl.compiler.codegen import CUDA_LIKE_BACKENDS, generate_wrappers_for_op
from dsl.compiler.registry import REGISTRY
from dsl.ops import discover


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
