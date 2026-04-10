"""Decorators for registering InfiniOps operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from dsl.compiler.registry import REGISTRY


@dataclass
class ManualOpDef:
    """An operator whose kernel logic is hand-written in C++."""

    name: str
    base: str
    backends: dict[str, str | dict[str, str]] = field(default_factory=dict)


@dataclass
class InfiniOpDef:
    """An operator whose CUDA/CPU kernels are auto-generated from DSL."""

    name: str
    shapes: dict[str, str] = field(default_factory=dict)
    manual_backends: dict[str, str] = field(default_factory=dict)
    func: Callable[..., Any] | None = None


def manual_op(
    *,
    name: str,
    base: str,
    backends: dict[str, str | dict[str, str]] | None = None,
) -> Callable:
    """Register a hand-written operator.

    The compiler generates only boilerplate (backend wrappers, bindings)
    while kernel logic stays in the files specified by ``backends``.
    """

    def decorator(func: Callable) -> ManualOpDef:
        op = ManualOpDef(
            name=name,
            base=base,
            backends=backends or {},
        )
        REGISTRY.register(op)

        return op

    return decorator


def infini_op(
    *,
    name: str,
    shapes: dict[str, str] | None = None,
    manual_backends: dict[str, str] | None = None,
) -> Callable:
    """Register an operator defined in the DSL.

    CUDA-like backends and CPU get auto-generated kernel code.
    Backends listed in ``manual_backends`` use the specified hand-written
    implementations instead.
    """

    def decorator(func: Callable) -> InfiniOpDef:
        op = InfiniOpDef(
            name=name,
            shapes=shapes or {},
            manual_backends=manual_backends or {},
            func=func,
        )
        REGISTRY.register(op)

        return op

    return decorator
