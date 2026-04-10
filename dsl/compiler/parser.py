"""Parse `@infini_op` function bodies into a compute DAG."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING, Any

from dsl.compiler.dag import ComputeDAG, NodeKind

if TYPE_CHECKING:
    from dsl.decorators import InfiniOpDef

# Map Python AST binary operators to DAG node kinds.
_BINOP_MAP: dict[type, NodeKind] = {
    ast.Add: NodeKind.ADD,
    ast.Sub: NodeKind.SUB,
    ast.Mult: NodeKind.MUL,
    ast.Div: NodeKind.DIV,
    ast.Pow: NodeKind.POW,
}

# Map Python AST comparison operators to DAG node kinds.
_CMPOP_MAP: dict[type, NodeKind] = {
    ast.Gt: NodeKind.GT,
    ast.Lt: NodeKind.LT,
    ast.GtE: NodeKind.GE,
    ast.LtE: NodeKind.LE,
    ast.Eq: NodeKind.EQ,
}

# Map DSL function names to DAG node kinds.
_FUNC_MAP: dict[str, NodeKind] = {
    "sqrt": NodeKind.SQRT,
    "rsqrt": NodeKind.RSQRT,
    "exp": NodeKind.EXP,
    "log": NodeKind.LOG,
    "abs": NodeKind.ABS,
    "neg": NodeKind.NEG,
    "relu": NodeKind.RELU,
    "gelu": NodeKind.GELU,
    "silu": NodeKind.SILU,
    "sigmoid": NodeKind.SIGMOID,
    "tanh": NodeKind.TANH,
    "reduce_sum": NodeKind.REDUCE_SUM,
    "reduce_mean": NodeKind.REDUCE_MEAN,
    "reduce_max": NodeKind.REDUCE_MAX,
    "reduce_min": NodeKind.REDUCE_MIN,
    "cast": NodeKind.CAST,
    "where": NodeKind.WHERE,
    "clamp": NodeKind.CLAMP,
}


class _DAGBuilder(ast.NodeVisitor):
    """Walk a function AST and build a ``ComputeDAG``."""

    def __init__(self, dag: ComputeDAG, params: dict[str, dict[str, Any]]) -> None:
        self.dag = dag
        self.params = params

        # Maps local variable names to DAG node ids.
        self.env: dict[str, int] = {}

        # Register function parameters as INPUT / SCALAR nodes.
        for pname, pinfo in params.items():

            if pinfo["kind"] == "tensor":
                nid = dag.add_node(
                    NodeKind.INPUT,
                    name=pname,
                    shape=pinfo.get("shape"),
                )
            else:
                nid = dag.add_node(NodeKind.SCALAR, name=pname)

            self.env[pname] = nid

    def visit_Assign(self, node: ast.Assign) -> None:
        assert len(node.targets) == 1, "Only single assignment supported."
        target = node.targets[0]
        assert isinstance(target, ast.Name)

        nid = self._visit_expr(node.value)
        self.env[target.id] = nid

    def visit_Return(self, node: ast.Return) -> None:
        assert node.value is not None
        nid = self._visit_expr(node.value)
        self.dag.output_id = nid

    def _visit_expr(self, node: ast.expr) -> int:
        """Recursively translate an expression AST node into DAG nodes."""

        if isinstance(node, ast.Name):
            assert node.id in self.env, f"Undefined variable: `{node.id}`."

            return self.env[node.id]

        if isinstance(node, ast.Constant):

            return self.dag.add_node(
                NodeKind.SCALAR,
                name=repr(node.value),
                attrs={"value": node.value},
            )

        if isinstance(node, ast.BinOp):

            return self._visit_binop(node)

        if isinstance(node, ast.UnaryOp):

            return self._visit_unaryop(node)

        if isinstance(node, ast.Call):

            return self._visit_call(node)

        if isinstance(node, ast.Compare):

            return self._visit_compare(node)

        raise ValueError(f"Unsupported expression type: {type(node).__name__}.")

    def _visit_binop(self, node: ast.BinOp) -> int:
        left = self._visit_expr(node.left)
        right = self._visit_expr(node.right)
        kind = _BINOP_MAP.get(type(node.op))

        if kind is None:
            raise ValueError(
                f"Unsupported binary operator: {type(node.op).__name__}."
            )

        return self.dag.add_node(kind, inputs=[left, right])

    def _visit_unaryop(self, node: ast.UnaryOp) -> int:
        operand = self._visit_expr(node.operand)

        if isinstance(node.op, ast.USub):

            return self.dag.add_node(NodeKind.NEG, inputs=[operand])

        raise ValueError(
            f"Unsupported unary operator: {type(node.op).__name__}."
        )

    def _visit_call(self, node: ast.Call) -> int:
        func_name = self._get_func_name(node)
        kind = _FUNC_MAP.get(func_name)

        if kind is None:
            raise ValueError(f"Unknown DSL primitive: `{func_name}`.")

        # Build input list from positional args.
        inputs = [self._visit_expr(arg) for arg in node.args]

        # Extract keyword arguments.
        kwargs: dict[str, Any] = {}

        for kw in node.keywords:
            assert kw.arg is not None

            if isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
            elif isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value
            elif isinstance(kw.value, ast.Name):
                kwargs[kw.arg] = kw.value.id

        # Handle reduction ops.
        if kind in (
            NodeKind.REDUCE_SUM,
            NodeKind.REDUCE_MEAN,
            NodeKind.REDUCE_MAX,
            NodeKind.REDUCE_MIN,
        ):

            return self.dag.add_node(
                kind,
                inputs=inputs,
                reduce_dim=kwargs.get("dim"),
            )

        # Handle cast.
        if kind == NodeKind.CAST:

            return self.dag.add_node(
                kind,
                inputs=inputs,
                cast_dtype=kwargs.get("dtype"),
            )

        # Handle where(cond, a, b).
        if kind == NodeKind.WHERE:
            assert len(inputs) == 3, "`where` requires 3 arguments."

            return self.dag.add_node(kind, inputs=inputs)

        # Handle clamp.
        if kind == NodeKind.CLAMP:

            return self.dag.add_node(
                kind,
                inputs=inputs,
                clamp_min=kwargs.get("min"),
                clamp_max=kwargs.get("max"),
            )

        # Unary / activation functions.
        return self.dag.add_node(kind, inputs=inputs)

    def _visit_compare(self, node: ast.Compare) -> int:
        assert len(node.ops) == 1, "Only single comparisons supported."
        assert len(node.comparators) == 1

        left = self._visit_expr(node.left)
        right = self._visit_expr(node.comparators[0])
        kind = _CMPOP_MAP.get(type(node.ops[0]))

        if kind is None:
            raise ValueError(
                f"Unsupported comparison: {type(node.ops[0]).__name__}."
            )

        return self.dag.add_node(kind, inputs=[left, right])

    @staticmethod
    def _get_func_name(node: ast.Call) -> str:

        if isinstance(node.func, ast.Name):
            return node.func.id

        if isinstance(node.func, ast.Attribute):
            return node.func.attr

        raise ValueError(f"Unsupported call target: {type(node.func).__name__}.")


def _extract_params(func_def: ast.FunctionDef) -> dict[str, dict[str, Any]]:
    """Extract parameter metadata from the function signature AST."""
    params: dict[str, dict[str, Any]] = {}

    for arg in func_def.args.args:
        pname = arg.arg
        annotation = arg.annotation
        pinfo: dict[str, Any] = {"kind": "tensor"}

        if annotation is not None:

            # Tensor["B", "H", "D"] → subscript with shape vars.
            if isinstance(annotation, ast.Subscript):

                if isinstance(annotation.value, ast.Name):

                    if annotation.value.id == "Scalar":
                        pinfo["kind"] = "scalar"
                    elif annotation.value.id == "Tensor":
                        # Extract shape variable names.
                        shape = _extract_shape_vars(annotation.slice)
                        pinfo["shape"] = shape

            elif isinstance(annotation, ast.Name):

                if annotation.id == "float":
                    pinfo["kind"] = "scalar"
                elif annotation.id == "int":
                    pinfo["kind"] = "scalar"

        params[pname] = pinfo

    return params


def _extract_shape_vars(node: ast.expr) -> list[str]:
    """Extract shape variable names from a Tensor subscript."""

    if isinstance(node, ast.Tuple):
        return [_const_str(elt) for elt in node.elts]

    return [_const_str(node)]


def _const_str(node: ast.expr) -> str:

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    raise ValueError(f"Expected string constant, got {type(node).__name__}.")


def parse_infini_op(op: InfiniOpDef) -> ComputeDAG:
    """Parse an `@infini_op` function into a ``ComputeDAG``."""
    assert op.func is not None, f"Operator `{op.name}` has no function body."

    source = inspect.getsource(op.func)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    # Find the function definition (skip the decorator).
    func_def: ast.FunctionDef | None = None

    for node in ast.walk(tree):

        if isinstance(node, ast.FunctionDef):
            func_def = node

            break

    assert func_def is not None, "No function definition found."

    params = _extract_params(func_def)
    dag = ComputeDAG(shape_vars=dict(op.shapes))
    builder = _DAGBuilder(dag, params)

    for stmt in func_def.body:
        builder.visit(stmt)

    assert dag.output_id is not None, (
        f"Operator `{op.name}` function body has no return statement."
    )

    return dag
