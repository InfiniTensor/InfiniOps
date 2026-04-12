"""Tests for the DSL compiler pipeline."""

from __future__ import annotations


from dsl.compiler.dag import NodeKind
from dsl.compiler.parser import parse_infini_op
from dsl.compiler.patterns import BrickKind, match_dag
from dsl.compiler.infini_codegen import generate_cuda_kernel, generate_cpu_kernel
from dsl.decorators import InfiniOpDef


# ---- Helpers ---------------------------------------------------------------


def _make_add_op() -> InfiniOpDef:
    """Create a simple binary add @infini_op."""

    def add_fn(input, other):
        return input + other

    return InfiniOpDef(
        name="TestAdd",
        shapes={"N": "output_size"},
        func=add_fn,
    )


def _make_rms_norm_op() -> InfiniOpDef:
    """Create an RmsNorm-like @infini_op."""

    def rms_norm_fn(input, weight, eps=1e-6):
        from dsl.primitives import reduce_mean, rsqrt

        ss = reduce_mean(input * input, dim="D")
        rms = rsqrt(ss + eps)

        return input * rms * weight

    return InfiniOpDef(
        name="TestRmsNorm",
        shapes={"B": "batch_size", "H": "nhead", "D": "dim"},
        func=rms_norm_fn,
    )


# ---- Parser tests ----------------------------------------------------------


class TestParser:
    def test_parse_add(self) -> None:
        op = _make_add_op()
        dag = parse_infini_op(op)

        assert dag.output_id is not None
        assert len(dag.nodes) > 0

        # Should have 2 inputs and 1 add.
        inputs = [n for n in dag.nodes.values() if n.kind == NodeKind.INPUT]
        adds = [n for n in dag.nodes.values() if n.kind == NodeKind.ADD]
        assert len(inputs) == 2
        assert len(adds) == 1

    def test_parse_rms_norm(self) -> None:
        op = _make_rms_norm_op()
        dag = parse_infini_op(op)

        assert dag.output_id is not None
        assert dag.has_reduction()

        reductions = dag.reduction_nodes()
        assert len(reductions) == 1
        assert reductions[0].kind == NodeKind.REDUCE_MEAN

    def test_elementwise_only(self) -> None:
        op = _make_add_op()
        dag = parse_infini_op(op)
        assert dag.is_elementwise_only()

    def test_topo_sort(self) -> None:
        op = _make_add_op()
        dag = parse_infini_op(op)
        topo = dag.topo_sort()

        # Output should be last.
        assert topo[-1] == dag.output_id


# ---- Pattern matching tests ------------------------------------------------


class TestPatterns:
    def test_match_add(self) -> None:
        op = _make_add_op()
        dag = parse_infini_op(op)
        result = match_dag(dag)

        assert result.brick == BrickKind.BINARY_ELEMENTWISE

    def test_match_rms_norm(self) -> None:
        op = _make_rms_norm_op()
        dag = parse_infini_op(op)
        result = match_dag(dag)

        assert result.brick == BrickKind.REDUCE_THEN_TRANSFORM
        assert result.reduce_nodes is not None
        assert result.transform_nodes is not None


# ---- Code generation tests ------------------------------------------------


class TestCodegen:
    def test_cuda_add(self) -> None:
        op = _make_add_op()
        dag = parse_infini_op(op)
        match = match_dag(dag)
        code = generate_cuda_kernel(op, dag, match)

        assert "#ifndef" in code
        assert "TestAddOp" in code
        assert "BinaryElementwiseBrick" in code
        assert "va + vb" in code

    def test_cpu_add(self) -> None:
        op = _make_add_op()
        dag = parse_infini_op(op)
        match = match_dag(dag)
        code = generate_cpu_kernel(op, dag, match)

        assert "#ifndef" in code
        assert "CpuTestAddOp" in code
        assert "CpuBinaryElementwise" in code

    def test_cuda_rms_norm(self) -> None:
        op = _make_rms_norm_op()
        dag = parse_infini_op(op)
        match = match_dag(dag)
        code = generate_cuda_kernel(op, dag, match)

        assert "TestRmsNormReduce" in code
        assert "TestRmsNormTransform" in code
        assert "LaunchReduceThenTransform" in code
        assert "rsqrtf" in code
        assert "epsilon" in code

    def test_cpu_rms_norm(self) -> None:
        op = _make_rms_norm_op()
        dag = parse_infini_op(op)
        match = match_dag(dag)
        code = generate_cpu_kernel(op, dag, match)

        assert "CpuTestRmsNormReduce" in code
        assert "CpuTestRmsNormTransform" in code
        assert "CpuReduceThenTransform" in code
        assert "std::sqrt" in code
