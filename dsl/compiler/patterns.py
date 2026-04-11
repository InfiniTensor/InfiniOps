"""Pattern matching: map compute DAG subgraphs to C++ template bricks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from dsl.compiler.dag import (
    ELEMENTWISE_BINARY,
    ELEMENTWISE_UNARY,
    ComputeDAG,
    DagNode,
    NodeKind,
)

if TYPE_CHECKING:
    pass


class BrickKind(Enum):
    """Available C++ template bricks."""

    BINARY_ELEMENTWISE = auto()
    UNARY_ELEMENTWISE = auto()
    REDUCE_THEN_TRANSFORM = auto()
    PURE_REDUCTION = auto()


@dataclass
class MatchResult:
    """Result of matching a compute DAG to a brick pattern."""

    brick: BrickKind

    # For REDUCE_THEN_TRANSFORM: the reduce and transform sub-DAGs.
    reduce_nodes: list[int] | None = None
    transform_nodes: list[int] | None = None
    reduce_dim: str | None = None

    # For elementwise: the functor body description.
    elementwise_kind: str | None = None

    # The input parameter names involved.
    input_names: list[str] | None = None


def match_dag(dag: ComputeDAG) -> MatchResult:
    """Match a compute DAG to the best-fitting brick pattern.

    Raises ``ValueError`` if no pattern matches.
    """

    if dag.is_elementwise_only():
        return _match_elementwise(dag)

    if dag.has_reduction():
        return _match_reduce_then_transform(dag)

    raise ValueError(
        "Cannot match DAG to any known brick pattern. "
        "Consider using `@manual_op` instead."
    )


def _match_elementwise(dag: ComputeDAG) -> MatchResult:
    """Match a pure-elementwise DAG."""

    # Collect input tensor names.
    inputs = [
        n.name
        for n in dag.nodes.values()
        if n.kind == NodeKind.INPUT and n.name is not None
    ]

    # Determine if it is a binary or unary elementwise op.
    compute_nodes = [
        n
        for n in dag.nodes.values()
        if n.kind not in (NodeKind.INPUT, NodeKind.SCALAR)
    ]

    # Count tensor inputs (not scalar).
    tensor_inputs = [
        n for n in dag.nodes.values() if n.kind == NodeKind.INPUT
    ]

    if len(tensor_inputs) >= 2:
        # Determine the core operation kind for simple binary ops.
        kind = _identify_core_op(dag, compute_nodes)

        return MatchResult(
            brick=BrickKind.BINARY_ELEMENTWISE,
            elementwise_kind=kind,
            input_names=inputs,
        )

    return MatchResult(
        brick=BrickKind.UNARY_ELEMENTWISE,
        elementwise_kind=_identify_core_op(dag, compute_nodes),
        input_names=inputs,
    )


def _match_reduce_then_transform(dag: ComputeDAG) -> MatchResult:
    """Match a reduce-then-transform pattern.

    The DAG must have exactly one reduction, followed by elementwise ops
    that use the reduction result.
    """
    reductions = dag.reduction_nodes()

    if not reductions:
        raise ValueError("Expected at least one reduction node.")

    # Use the first reduction as the primary one.
    reduce_node = reductions[0]

    # Identify all nodes that contribute to the reduction (pre-reduce).
    reduce_ancestors = _ancestors(dag, reduce_node.id)
    reduce_ancestors.add(reduce_node.id)

    # Everything after the reduction is the transform.
    topo = dag.topo_sort()
    reduce_idx = topo.index(reduce_node.id)
    transform_ids = [
        nid
        for nid in topo[reduce_idx + 1 :]
        if dag.get(nid).kind not in (NodeKind.INPUT, NodeKind.SCALAR)
    ]

    # Collect input names.
    inputs = [
        n.name
        for n in dag.nodes.values()
        if n.kind == NodeKind.INPUT and n.name is not None
    ]

    return MatchResult(
        brick=BrickKind.REDUCE_THEN_TRANSFORM,
        reduce_nodes=sorted(reduce_ancestors),
        transform_nodes=transform_ids,
        reduce_dim=reduce_node.reduce_dim,
        input_names=inputs,
    )


def _ancestors(dag: ComputeDAG, nid: int) -> set[int]:
    """Return all ancestor node ids (transitive inputs), excluding leaf nodes."""
    result: set[int] = set()
    stack = list(dag.get(nid).inputs)

    while stack:
        cur = stack.pop()
        node = dag.get(cur)

        if node.kind in (NodeKind.INPUT, NodeKind.SCALAR):
            continue

        if cur not in result:
            result.add(cur)
            stack.extend(node.inputs)

    return result


def _identify_core_op(dag: ComputeDAG, compute_nodes: list[DagNode]) -> str:
    """Identify the dominant operation kind for simple elementwise DAGs."""

    if len(compute_nodes) == 1:
        return compute_nodes[0].kind.name.lower()

    # For compound expressions, return a description.
    kinds = {n.kind for n in compute_nodes}

    if kinds <= ELEMENTWISE_BINARY:
        return "compound_binary"

    if kinds <= ELEMENTWISE_UNARY:
        return "compound_unary"

    return "compound_mixed"
