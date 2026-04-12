"""Compute DAG representation for `@infini_op` operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class NodeKind(Enum):
    """Primitive operation types in the compute DAG."""

    # Inputs.
    INPUT = auto()
    SCALAR = auto()

    # Elementwise unary.
    NEG = auto()
    ABS = auto()
    SQRT = auto()
    RSQRT = auto()
    EXP = auto()
    LOG = auto()

    # Elementwise binary.
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    POW = auto()

    # Activations.
    RELU = auto()
    GELU = auto()
    SILU = auto()
    SIGMOID = auto()
    TANH = auto()

    # Reductions.
    REDUCE_SUM = auto()
    REDUCE_MEAN = auto()
    REDUCE_MAX = auto()
    REDUCE_MIN = auto()

    # Comparison / conditional.
    WHERE = auto()
    GT = auto()
    LT = auto()
    GE = auto()
    LE = auto()
    EQ = auto()

    # Type.
    CAST = auto()

    # Clamp.
    CLAMP = auto()


# Classify node kinds into categories for pattern matching.
ELEMENTWISE_UNARY = {
    NodeKind.NEG,
    NodeKind.ABS,
    NodeKind.SQRT,
    NodeKind.RSQRT,
    NodeKind.EXP,
    NodeKind.LOG,
    NodeKind.RELU,
    NodeKind.GELU,
    NodeKind.SILU,
    NodeKind.SIGMOID,
    NodeKind.TANH,
}

ELEMENTWISE_BINARY = {
    NodeKind.ADD,
    NodeKind.SUB,
    NodeKind.MUL,
    NodeKind.DIV,
    NodeKind.POW,
    NodeKind.GT,
    NodeKind.LT,
    NodeKind.GE,
    NodeKind.LE,
    NodeKind.EQ,
}

ELEMENTWISE = ELEMENTWISE_UNARY | ELEMENTWISE_BINARY | {
    NodeKind.WHERE,
    NodeKind.CAST,
    NodeKind.CLAMP,
}

REDUCTIONS = {
    NodeKind.REDUCE_SUM,
    NodeKind.REDUCE_MEAN,
    NodeKind.REDUCE_MAX,
    NodeKind.REDUCE_MIN,
}


@dataclass
class DagNode:
    """A single node in the compute DAG."""

    id: int
    kind: NodeKind
    inputs: list[int] = field(default_factory=list)

    # Shape variable name (e.g. "B", "H", "D") for inputs.
    shape: list[str] | None = None

    # For INPUT: parameter name; for SCALAR: value/name.
    name: str | None = None

    # For reductions: the shape variable being reduced over.
    reduce_dim: str | None = None

    # For CAST: target dtype string.
    cast_dtype: str | None = None

    # For CLAMP: min/max bounds.
    clamp_min: float | None = None
    clamp_max: float | None = None

    # Arbitrary extra attributes.
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputeDAG:
    """A directed acyclic graph of primitive operations.

    Built by the parser from an `@infini_op` function body.
    """

    nodes: dict[int, DagNode] = field(default_factory=dict)
    output_id: int | None = None

    # Shape variables declared in the operator definition.
    shape_vars: dict[str, str] = field(default_factory=dict)

    _next_id: int = field(default=0, repr=False)

    def add_node(self, kind: NodeKind, **kwargs: Any) -> int:
        """Create a new node and return its id."""
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = DagNode(id=nid, kind=kind, **kwargs)

        return nid

    def get(self, nid: int) -> DagNode:
        return self.nodes[nid]

    def consumers(self, nid: int) -> list[int]:
        """Return ids of nodes that consume ``nid`` as an input."""

        return [
            n.id for n in self.nodes.values() if nid in n.inputs
        ]

    def is_elementwise_only(self) -> bool:
        """True if the DAG contains only elementwise ops (no reductions)."""

        for node in self.nodes.values():

            if node.kind in REDUCTIONS:
                return False

        return True

    def has_reduction(self) -> bool:
        """True if any node is a reduction."""

        return any(n.kind in REDUCTIONS for n in self.nodes.values())

    def reduction_nodes(self) -> list[DagNode]:
        """Return all reduction nodes."""

        return [n for n in self.nodes.values() if n.kind in REDUCTIONS]

    def topo_sort(self) -> list[int]:
        """Return node ids in topological order."""
        visited: set[int] = set()
        order: list[int] = []

        def dfs(nid: int) -> None:

            if nid in visited:
                return

            visited.add(nid)

            for inp in self.nodes[nid].inputs:
                dfs(inp)

            order.append(nid)

        for nid in self.nodes:
            dfs(nid)

        return order
