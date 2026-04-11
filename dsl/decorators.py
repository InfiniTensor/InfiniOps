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
    impl_names: dict[int, str] = field(default_factory=dict)


@dataclass
class InfiniOpDef:
    """An operator whose CUDA/CPU kernels are auto-generated from DSL."""

    name: str
    shapes: dict[str, str] = field(default_factory=dict)
    manual_backends: dict[str, str] = field(default_factory=dict)
    func: Callable[..., Any] | None = None
    impl_index: int = 0


def manual_op(
    *,
    name: str,
    base: str,
    backends: dict[str, str | dict[str, str]] | None = None,
    impl_names: dict[int, str] | None = None,
) -> Callable:
    """Register a hand-written operator.

    The compiler generates only boilerplate (backend wrappers, bindings)
    while kernel logic stays in the files specified by ``backends``.

    ``impl_names`` maps implementation indices to human-readable names
    (e.g. ``{0: "cublas", 1: "cublaslt"}``).  When omitted, the default
    mapping ``{0: "default"}`` is used.
    """

    def decorator(func: Callable) -> ManualOpDef:
        op = ManualOpDef(
            name=name,
            base=base,
            backends=backends or {},
            impl_names=impl_names or {},
        )
        REGISTRY.register(op)

        return op

    return decorator


def infini_op(
    *,
    name: str,
    shapes: dict[str, str] | None = None,
    manual_backends: dict[str, str] | None = None,
    impl_index: int = 0,
) -> Callable:
    """Register an operator defined in the DSL.

    CUDA-like backends and CPU get auto-generated kernel code.
    Backends listed in ``manual_backends`` use the specified hand-written
    implementations instead.

    When ``impl_index > 0``, the operator is registered as an alternative
    implementation of an existing operator (like cuBLAS vs cuBLASLt for
    GEMM).  The compiler generates ``Operator<Op, kDev, impl_index>``
    specializations and a ``registry.h`` declaring ``List<0, ..., N>``.
    """

    def decorator(func: Callable) -> InfiniOpDef:
        op = InfiniOpDef(
            name=name,
            shapes=shapes or {},
            manual_backends=manual_backends or {},
            func=func,
            impl_index=impl_index,
        )
        REGISTRY.register(op)

        return op

    return decorator
