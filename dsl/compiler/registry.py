"""Global registry collecting all operator definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dsl.decorators import InfiniOpDef, ManualOpDef


class _Registry:
    def __init__(self) -> None:
        self._ops: dict[str, ManualOpDef | InfiniOpDef] = {}

    def register(self, op: ManualOpDef | InfiniOpDef) -> None:
        if op.name in self._ops:
            raise ValueError(f"Operator `{op.name}` is already registered.")

        self._ops[op.name] = op

    def get(self, name: str) -> ManualOpDef | InfiniOpDef:
        return self._ops[name]

    def all_ops(self) -> dict[str, ManualOpDef | InfiniOpDef]:
        return dict(self._ops)

    def clear(self) -> None:
        self._ops.clear()


REGISTRY = _Registry()
