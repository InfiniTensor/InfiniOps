"""Operator definitions for InfiniOps.

Importing this package auto-discovers and registers all operator definitions
in this directory.
"""

import importlib
import pathlib

_OPS_DIR = pathlib.Path(__file__).parent


def discover() -> None:
    """Import every Python module in this package to trigger registration."""

    for path in sorted(_OPS_DIR.glob("*.py")):

        if path.name.startswith("_"):
            continue

        module_name = f"dsl.ops.{path.stem}"
        importlib.import_module(module_name)
