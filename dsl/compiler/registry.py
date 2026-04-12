"""Global registry collecting all operator definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dsl.decorators import InfiniOpDef, ManualOpDef


class _Registry:
    def __init__(self) -> None:
        self._ops: dict[str, ManualOpDef | InfiniOpDef] = {}
        self._variants: dict[str, list[InfiniOpDef]] = {}

    def register(self, op: ManualOpDef | InfiniOpDef) -> None:
        from dsl.decorators import InfiniOpDef

        if isinstance(op, InfiniOpDef) and op.impl_index > 0:
            self._variants.setdefault(op.name, []).append(op)

            return

        if op.name in self._ops:
            raise ValueError(f"Operator `{op.name}` is already registered.")

        self._ops[op.name] = op

    def get(self, name: str) -> ManualOpDef | InfiniOpDef:
        return self._ops[name]

    def all_ops(self) -> dict[str, ManualOpDef | InfiniOpDef]:
        return dict(self._ops)

    def variants(self, name: str) -> list[InfiniOpDef]:
        """Return DSL alternative implementations for a given operator."""

        return list(self._variants.get(name, []))

    def all_variants(self) -> dict[str, list[InfiniOpDef]]:
        """Return all DSL variant implementations."""

        return dict(self._variants)

    def impl_names_for(self, name: str) -> dict[str, int]:
        """Return the merged name→index mapping for an operator.

        Rules:
        - ``@manual_op`` with explicit ``impl_names`` → use as-is.
        - ``@manual_op`` without ``impl_names`` → ``{"default": 0}``.
        - Each ``@infini_op`` variant adds ``{"dsl": impl_index}``.
        """
        from dsl.decorators import ManualOpDef

        primary = self._ops.get(name)
        result: dict[str, int] = {}

        if primary is not None:

            if isinstance(primary, ManualOpDef) and primary.impl_names:
                result = {v: k for k, v in primary.impl_names.items()}
            else:
                result = {"default": 0}

        for variant in self._variants.get(name, []):
            result["dsl"] = variant.impl_index

        return result

    def all_impl_names(self) -> dict[str, dict[str, int]]:
        """Return name→index mappings for all operators."""
        all_names = set(self._ops.keys()) | set(self._variants.keys())

        return {name: self.impl_names_for(name) for name in sorted(all_names)}

    def clear(self) -> None:
        self._ops.clear()
        self._variants.clear()


REGISTRY = _Registry()
