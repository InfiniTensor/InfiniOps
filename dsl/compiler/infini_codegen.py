"""C++ code generation for `@infini_op` operators.

Translates a matched compute DAG into C++ source files that compose
template bricks from `src/cuda/templates/` and `src/cpu/templates/`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dsl.compiler.dag import ComputeDAG, DagNode, NodeKind
from dsl.compiler.patterns import BrickKind, MatchResult

if TYPE_CHECKING:
    from dsl.decorators import InfiniOpDef


def _to_snake(pascal: str) -> str:
    """Convert PascalCase to snake_case."""
    import re

    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", pascal).lower()


# ---- Functor C++ expression generation -------------------------------------


# Map DAG node kinds to C++ operator/function expressions.
_CUDA_BINOP: dict[NodeKind, str] = {
    NodeKind.ADD: "+",
    NodeKind.SUB: "-",
    NodeKind.MUL: "*",
    NodeKind.DIV: "/",
}

_CUDA_UNARY_FUNC: dict[NodeKind, str] = {
    NodeKind.SQRT: "sqrtf",
    NodeKind.RSQRT: "rsqrtf",
    NodeKind.EXP: "expf",
    NodeKind.LOG: "logf",
    NodeKind.ABS: "fabsf",
    NodeKind.TANH: "tanhf",
}

_CPU_UNARY_FUNC: dict[NodeKind, str] = {
    NodeKind.SQRT: "std::sqrt",
    NodeKind.RSQRT: "1.f / std::sqrt",
    NodeKind.EXP: "std::exp",
    NodeKind.LOG: "std::log",
    NodeKind.ABS: "std::abs",
    NodeKind.TANH: "std::tanh",
}

_ACTIVATION_CUDA: dict[NodeKind, str] = {
    NodeKind.RELU: "v > 0 ? v : static_cast<ComputeType>(0)",
    NodeKind.SIGMOID: "static_cast<ComputeType>(1) / (static_cast<ComputeType>(1) + expf(-v))",
    NodeKind.SILU: "v / (static_cast<ComputeType>(1) + expf(-v))",
}

_ACTIVATION_CPU: dict[NodeKind, str] = {
    NodeKind.RELU: "v > 0 ? v : static_cast<ComputeType>(0)",
    NodeKind.SIGMOID: "static_cast<ComputeType>(1) / (static_cast<ComputeType>(1) + std::exp(-v))",
    NodeKind.SILU: "v / (static_cast<ComputeType>(1) + std::exp(-v))",
}


def _expr_for_node(
    dag: ComputeDAG,
    node: DagNode,
    var_map: dict[int, str],
    is_cuda: bool,
) -> str:
    """Generate a C++ expression string for a single DAG node.

    ``var_map`` maps node id → C++ variable name for already-emitted nodes.
    """

    def _ref(nid: int) -> str:
        return var_map[nid]

    if node.kind in _CUDA_BINOP:
        op = _CUDA_BINOP[node.kind]

        return f"({_ref(node.inputs[0])} {op} {_ref(node.inputs[1])})"

    unary_map = _CUDA_UNARY_FUNC if is_cuda else _CPU_UNARY_FUNC

    if node.kind in unary_map:
        func = unary_map[node.kind]

        if node.kind == NodeKind.RSQRT and not is_cuda:
            return f"(1.f / std::sqrt({_ref(node.inputs[0])}))"

        return f"{func}({_ref(node.inputs[0])})"

    if node.kind == NodeKind.NEG:
        return f"(-{_ref(node.inputs[0])})"

    act_map = _ACTIVATION_CUDA if is_cuda else _ACTIVATION_CPU

    if node.kind in act_map:
        # Activation functions expect the variable to be named `v`.
        return act_map[node.kind].replace("v", _ref(node.inputs[0]))

    if node.kind == NodeKind.WHERE:
        return (
            f"({_ref(node.inputs[0])} ? "
            f"{_ref(node.inputs[1])} : {_ref(node.inputs[2])})"
        )

    if node.kind == NodeKind.POW:
        func = "powf" if is_cuda else "std::pow"

        return f"{func}({_ref(node.inputs[0])}, {_ref(node.inputs[1])})"

    if node.kind == NodeKind.SCALAR:
        # Literal scalar.
        val = node.attrs.get("value")

        if val is not None:
            return repr(val)

        return node.name or "0"

    raise ValueError(f"Cannot generate expression for node kind: {node.kind}.")


# ---- Binary elementwise code generation ------------------------------------


def _dsl_prefix(op: InfiniOpDef) -> str:
    """Return the prefix for DSL-generated class names.

    When ``impl_index > 0``, class names are prefixed with ``Dsl`` to
    avoid collisions with the hand-written implementation.
    """

    return "Dsl" if op.impl_index > 0 else ""


def _generate_binary_functor_cuda(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
) -> str:
    """Generate the device-side binary functor for CUDA."""
    prefix = _dsl_prefix(op)

    # Build the functor body by walking the DAG in topological order.
    topo = dag.topo_sort()
    var_map: dict[int, str] = {}
    body_lines: list[str] = []

    for nid in topo:
        node = dag.get(nid)

        if node.kind == NodeKind.INPUT:

            if node.name == match.input_names[0]:
                var_map[nid] = "va"
            elif node.name == match.input_names[1]:
                var_map[nid] = "vb"
            else:
                var_map[nid] = node.name

            continue

        if node.kind == NodeKind.SCALAR:
            val = node.attrs.get("value")

            if val is not None:
                var_map[nid] = repr(val)
            else:
                var_map[nid] = node.name

            continue

        expr = _expr_for_node(dag, node, var_map, is_cuda=True)

        if nid == dag.output_id:
            body_lines.append(f"    return Caster<kDev>::template Cast<T>({expr});")
        else:
            vname = f"t{nid}"
            body_lines.append(f"    auto {vname} = {expr};")
            var_map[nid] = vname

    body = "\n".join(body_lines)
    functor_name = f"{prefix}{op.name}Op"

    return f"""\
// Device-side binary functor for `{op.name}` (DSL).
template <Device::Type kDev>
struct {functor_name} {{
  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {{
    using ComputeType = float;
    auto va = Caster<kDev>::template Cast<ComputeType>(a);
    auto vb = Caster<kDev>::template Cast<ComputeType>(b);
{body}
  }}
}};"""


def _generate_binary_functor_cpu(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
) -> str:
    """Generate the host-side binary functor for CPU."""
    prefix = _dsl_prefix(op)
    topo = dag.topo_sort()
    var_map: dict[int, str] = {}
    body_lines: list[str] = []

    for nid in topo:
        node = dag.get(nid)

        if node.kind == NodeKind.INPUT:

            if node.name == match.input_names[0]:
                var_map[nid] = "va"
            elif node.name == match.input_names[1]:
                var_map[nid] = "vb"
            else:
                var_map[nid] = node.name

            continue

        if node.kind == NodeKind.SCALAR:
            val = node.attrs.get("value")

            if val is not None:
                var_map[nid] = repr(val)
            else:
                var_map[nid] = node.name

            continue

        expr = _expr_for_node(dag, node, var_map, is_cuda=False)

        if nid == dag.output_id:
            body_lines.append(f"    return static_cast<T>({expr});")
        else:
            vname = f"t{nid}"
            body_lines.append(f"    auto {vname} = {expr};")
            var_map[nid] = vname

    body = "\n".join(body_lines)
    functor_name = f"{prefix}Cpu{op.name}Op"

    return f"""\
// Host-side binary functor for `{op.name}` (CPU, DSL).
struct {functor_name} {{
  template <typename T>
  T operator()(const T& a, const T& b) const {{
    using ComputeType = float;
    auto va = static_cast<ComputeType>(a);
    auto vb = static_cast<ComputeType>(b);
{body}
  }}
}};"""


# ---- Reduce-then-transform code generation ---------------------------------


def _generate_reduce_op_cuda(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
) -> str:
    """Generate the CUDA reduce op struct."""
    assert match.reduce_nodes is not None

    # Analyze the reduction pattern to determine the accumulation.
    reduce_node = None

    for nid in match.reduce_nodes:
        node = dag.get(nid)

        if node.kind in (
            NodeKind.REDUCE_SUM,
            NodeKind.REDUCE_MEAN,
            NodeKind.REDUCE_MAX,
            NodeKind.REDUCE_MIN,
        ):
            reduce_node = node

            break

    assert reduce_node is not None

    # Determine the pre-reduce expression (what is accumulated).
    pre_reduce_expr = _build_pre_reduce_expr(dag, reduce_node, is_cuda=True)
    finalize_expr = _build_finalize_expr(dag, reduce_node, match, is_cuda=True)

    prefix = _dsl_prefix(op)

    return f"""\
// Reduce op for `{op.name}` (DSL).
struct {prefix}{op.name}Reduce {{
  template <unsigned int block_size, Device::Type kDev, typename TData>
  __device__ __forceinline__ float Accumulate(const TData* ptr,
                                              size_t count) const {{
    float ss = 0;

    for (size_t i = threadIdx.x; i < count; i += block_size) {{
      float v = Caster<kDev>::template Cast<float>(ptr[i]);
{pre_reduce_expr}
    }}

    return ss;
  }}

  __device__ __forceinline__ float Finalize(float total,
                                            size_t count) const {{
{finalize_expr}
  }}

{_generate_reduce_members(op, dag, match)}
}};"""


def _generate_reduce_op_cpu(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
) -> str:
    """Generate the CPU reduce op struct."""
    assert match.reduce_nodes is not None

    reduce_node = None

    for nid in match.reduce_nodes:
        node = dag.get(nid)

        if node.kind in (
            NodeKind.REDUCE_SUM,
            NodeKind.REDUCE_MEAN,
            NodeKind.REDUCE_MAX,
            NodeKind.REDUCE_MIN,
        ):
            reduce_node = node

            break

    assert reduce_node is not None

    init_val = _reduce_init_value(reduce_node.kind)
    accum_expr = _build_accum_expr_scalar(dag, reduce_node, is_cuda=False)
    finalize_expr = _build_finalize_expr(dag, reduce_node, match, is_cuda=False)

    prefix = _dsl_prefix(op)

    return f"""\
// CPU reduce op for `{op.name}` (DSL).
struct {prefix}Cpu{op.name}Reduce {{
  float Init() const {{ return {init_val}; }}

  float Accumulate(float acc, float v) const {{ return {accum_expr}; }}

  float Finalize(float acc, size_t count) const {{
{finalize_expr}
  }}

{_generate_reduce_members(op, dag, match)}
}};"""


def _generate_transform_op_cuda(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
) -> str:
    """Generate the CUDA transform op struct."""
    transform_body = _build_transform_body(dag, match, is_cuda=True)

    prefix = _dsl_prefix(op)

    return f"""\
// Transform op for `{op.name}` (DSL).
struct {prefix}{op.name}Transform {{
  template <Device::Type kDev, typename TData>
  __device__ __forceinline__ TData Apply(TData x, float reduced,
                                         size_t i) const {{
{transform_body}
  }}

{_generate_transform_members(op, dag, match)}
}};"""


def _generate_transform_op_cpu(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
) -> str:
    """Generate the CPU transform op struct."""
    transform_body = _build_transform_body(dag, match, is_cuda=False)

    prefix = _dsl_prefix(op)

    return f"""\
// CPU transform op for `{op.name}` (DSL).
struct {prefix}Cpu{op.name}Transform {{
  template <typename T>
  T Apply(T x, float reduced, size_t i) const {{
{transform_body}
  }}

{_generate_transform_members(op, dag, match)}
}};"""


# ---- Helper functions for reduce/transform expression building -------------


def _build_pre_reduce_expr(
    dag: ComputeDAG,
    reduce_node: DagNode,
    is_cuda: bool,
) -> str:
    """Build the inner-loop accumulation expression for the reduce phase."""

    # Walk the inputs to the reduction to find what is being accumulated.
    input_node_id = reduce_node.inputs[0]
    input_node = dag.get(input_node_id)

    # Common pattern: reduce_mean(x * x) → sum of squares.
    if (
        input_node.kind == NodeKind.MUL
        and len(input_node.inputs) == 2
        and input_node.inputs[0] == input_node.inputs[1]
    ):
        return "      ss += v * v;"

    # reduce_sum(x) or reduce_mean(x).
    if input_node.kind == NodeKind.INPUT:
        return "      ss += v;"

    # Generic: just accumulate the expression.
    var_map = {input_node.inputs[0]: "v"} if input_node.inputs else {"v": "v"}

    return "      ss += v;"


def _build_accum_expr_scalar(
    dag: ComputeDAG,
    reduce_node: DagNode,
    is_cuda: bool,
) -> str:
    """Build the scalar accumulation expression for CPU reduce."""
    input_node_id = reduce_node.inputs[0]
    input_node = dag.get(input_node_id)

    # reduce_mean(x * x) → acc + v * v.
    if (
        input_node.kind == NodeKind.MUL
        and len(input_node.inputs) == 2
        and input_node.inputs[0] == input_node.inputs[1]
    ):
        return "acc + v * v"

    return "acc + v"


def _reduce_init_value(kind: NodeKind) -> str:
    """Return the identity element for a reduction."""

    if kind in (NodeKind.REDUCE_SUM, NodeKind.REDUCE_MEAN):
        return "0.f"

    if kind == NodeKind.REDUCE_MAX:
        return "-INFINITY"

    if kind == NodeKind.REDUCE_MIN:
        return "INFINITY"

    return "0.f"


def _build_finalize_expr(
    dag: ComputeDAG,
    reduce_node: DagNode,
    match: MatchResult,
    is_cuda: bool,
) -> str:
    """Build the finalize expression after block reduction."""

    # Check what happens after the reduction before the transform phase.
    # Walk from the reduction output to find post-reduce ops.
    consumers = dag.consumers(reduce_node.id)
    topo = dag.topo_sort()

    # Find nodes between reduce and the first transform node.
    reduce_idx = topo.index(reduce_node.id)
    transform_start = (
        match.transform_nodes[0] if match.transform_nodes else dag.output_id
    )

    # Collect post-reduce nodes that are not transform nodes.
    post_reduce: list[int] = []

    for nid in topo[reduce_idx + 1 :]:

        if match.transform_nodes and nid in match.transform_nodes:
            break

        node = dag.get(nid)

        if node.kind not in (NodeKind.INPUT, NodeKind.SCALAR):
            post_reduce.append(nid)

    # Common pattern: rsqrt(total / count + eps).
    if reduce_node.kind == NodeKind.REDUCE_MEAN:
        # Check for rsqrt(mean + eps) pattern in post_reduce or transform.
        all_post = post_reduce + (match.transform_nodes or [])

        for nid in all_post:
            node = dag.get(nid)

            if node.kind == NodeKind.RSQRT:
                rsqrt_func = "rsqrtf" if is_cuda else "1.f / std::sqrt"

                if is_cuda:
                    return (
                        "    return rsqrtf(total / "
                        "static_cast<float>(count) + epsilon);"
                    )

                return (
                    "    return 1.f / std::sqrt(acc / "
                    "static_cast<float>(count) + epsilon);"
                )

        # Plain mean.
        if is_cuda:
            return "    return total / static_cast<float>(count);"

        return "    return acc / static_cast<float>(count);"

    if reduce_node.kind == NodeKind.REDUCE_SUM:

        if is_cuda:
            return "    return total;"

        return "    return acc;"

    if reduce_node.kind == NodeKind.REDUCE_MAX:

        if is_cuda:
            return "    return total;"

        return "    return acc;"

    if is_cuda:
        return "    return total;"

    return "    return acc;"


def _build_transform_body(
    dag: ComputeDAG,
    match: MatchResult,
    is_cuda: bool,
) -> str:
    """Build the transform phase body."""

    # The transform applies: out[i] = f(in[i], reduced, i).
    # Walk the DAG from the output backwards to understand the transform.
    output_node = dag.get(dag.output_id)

    # Common pattern: input * reduced * weight[i].
    # For RmsNorm: return x * rms * weight[i].
    if _is_rms_norm_transform(dag, match):

        if is_cuda:
            return (
                "    return Caster<kDev>::template Cast<TData>(\n"
                "        Caster<kDev>::template Cast<float>(x) *\n"
                "        Caster<kDev>::template Cast<float>("
                "static_cast<const TData*>(weight)[i]) * reduced);"
            )

        return (
            "    const auto* w = static_cast<const T*>(weight);\n\n"
            "    return Caster<Device::Type::kCpu>::Cast<T>(\n"
            "        Caster<Device::Type::kCpu>::Cast<float>(x) *\n"
            "        Caster<Device::Type::kCpu>::Cast<float>(w[i]) "
            "* reduced);"
        )

    # Generic: input * reduced.
    if is_cuda:
        return (
            "    return Caster<kDev>::template Cast<TData>(\n"
            "        Caster<kDev>::template Cast<float>(x) * reduced);"
        )

    return (
        "    return Caster<Device::Type::kCpu>::Cast<T>(\n"
        "        Caster<Device::Type::kCpu>::Cast<float>(x) * reduced);"
    )


def _is_rms_norm_transform(dag: ComputeDAG, match: MatchResult) -> bool:
    """Check if the transform is ``x * reduced * weight[i]``."""

    # Look for a weight tensor input.
    for node in dag.nodes.values():

        if node.kind == NodeKind.INPUT and node.name == "weight":
            return True

    return False


def _generate_reduce_members(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
) -> str:
    """Generate member variables for the reduce op struct."""
    members = []

    # Check if epsilon is used.
    for node in dag.nodes.values():

        if node.kind == NodeKind.SCALAR and node.name == "eps":
            members.append("  float epsilon;")

    return "\n".join(members)


def _generate_transform_members(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
) -> str:
    """Generate member variables for the transform op struct."""
    members = []

    for node in dag.nodes.values():

        if node.kind == NodeKind.INPUT and node.name == "weight":
            members.append("  const void* weight;")

    return "\n".join(members)


# ---- Top-level file generators ---------------------------------------------


def generate_cuda_kernel(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
) -> str:
    """Generate the shared CUDA kernel header for an `@infini_op`."""
    op_snake = _to_snake(op.name)

    if op.impl_index > 0:
        guard = f"INFINI_OPS_CUDA_{op_snake.upper()}_DSL_H_"
    else:
        guard = f"INFINI_OPS_CUDA_{op_snake.upper()}_KERNEL_H_"

    if match.brick == BrickKind.BINARY_ELEMENTWISE:
        return _gen_binary_elementwise_cuda(op, dag, match, guard, op_snake)

    if match.brick == BrickKind.REDUCE_THEN_TRANSFORM:
        return _gen_reduce_transform_cuda(op, dag, match, guard, op_snake)

    raise ValueError(f"Unsupported brick kind for CUDA codegen: {match.brick}.")


def generate_cpu_kernel(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
) -> str:
    """Generate the CPU implementation header for an `@infini_op`."""
    op_snake = _to_snake(op.name)

    if op.impl_index > 0:
        guard = f"INFINI_OPS_CPU_{op_snake.upper()}_DSL_H_"
    else:
        guard = f"INFINI_OPS_CPU_{op_snake.upper()}_{op_snake.upper()}_H_"

    if match.brick == BrickKind.BINARY_ELEMENTWISE:
        return _gen_binary_elementwise_cpu(op, dag, match, guard, op_snake)

    if match.brick == BrickKind.REDUCE_THEN_TRANSFORM:
        return _gen_reduce_transform_cpu(op, dag, match, guard, op_snake)

    raise ValueError(f"Unsupported brick kind for CPU codegen: {match.brick}.")


# ---- Binary elementwise file generators ------------------------------------


def _gen_binary_elementwise_cuda(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
    guard: str,
    op_snake: str,
) -> str:
    prefix = _dsl_prefix(op)
    functor = _generate_binary_functor_cuda(op, dag, match)
    base_header = f"base/{op_snake}.h"
    class_name = f"{prefix}Cuda{op.name}"
    functor_name = f"{prefix}{op.name}Op"

    return f"""\
#ifndef {guard}
#define {guard}

#include "cuda/templates/binary_elementwise.cuh"
#include "{base_header}"

namespace infini::ops {{

{functor}

template <typename Backend>
class {class_name} : public {op.name} {{
 public:
  {class_name}(const Tensor input, const Tensor other, Tensor out)
      : {op.name}{{input, other, out}},
        brick_{{input, other, out, ndim_}} {{}}

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {{
    brick_.template Run<AllTypes, {functor_name}>(
        stream_, input, other, out, output_size_, ndim_,
        is_input_contiguous_, is_other_contiguous_, is_out_contiguous_,
        out_type_);
  }}

 private:
  BinaryElementwiseBrick<Backend> brick_;
}};

}}  // namespace infini::ops

#endif
"""


def _gen_binary_elementwise_cpu(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
    guard: str,
    op_snake: str,
) -> str:
    prefix = _dsl_prefix(op)
    functor = _generate_binary_functor_cpu(op, dag, match)
    base_header = f"base/{op_snake}.h"
    functor_name = f"{prefix}Cpu{op.name}Op"
    impl_suffix = ", Impl::kDsl" if op.impl_index > 0 else ""
    impl_include = (
        f'#include "impl.h"\n#include "cpu/{op_snake}/registry.h"\n'
        if op.impl_index > 0
        else ""
    )

    return f"""\
#ifndef {guard}
#define {guard}

#include "cpu/templates/binary_elementwise.h"
#include "{base_header}"
{impl_include}
namespace infini::ops {{

{functor}

template <>
class Operator<{op.name}, Device::Type::kCpu{impl_suffix}> : public {op.name} {{
 public:
  using {op.name}::{op.name};

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {{
    CpuBinaryElementwise<AllTypes>(
        input, other, out, output_size_, ndim_,
        is_input_contiguous_, is_other_contiguous_, is_out_contiguous_,
        input_shape_, other_shape_, out_shape_,
        input_strides_, other_strides_, out_strides_,
        out_type_, {functor_name}{{}});
  }}
}};

}}  // namespace infini::ops

#endif
"""


# ---- Reduce-then-transform file generators ---------------------------------


def _gen_reduce_transform_cuda(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
    guard: str,
    op_snake: str,
) -> str:
    prefix = _dsl_prefix(op)
    reduce_op = _generate_reduce_op_cuda(op, dag, match)
    transform_op = _generate_transform_op_cuda(op, dag, match)
    base_header = f"base/{op_snake}.h"
    class_name = f"{prefix}Cuda{op.name}"
    reduce_name = f"{prefix}{op.name}Reduce"
    transform_name = f"{prefix}{op.name}Transform"

    # Determine the type list based on the operator.
    type_list = "ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>"

    return f"""\
#ifndef {guard}
#define {guard}

#include "cuda/templates/reduce_transform.cuh"
#include "{base_header}"

namespace infini::ops {{

{reduce_op}

{transform_op}

template <typename Backend>
class {class_name} : public {op.name} {{
 public:
  using {op.name}::{op.name};

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {{
    LaunchReduceThenTransform<Backend, {type_list}>(
        stream_, input, out, batch_size_, nhead_, dim_,
        out.dtype(), input_strides_, out_strides_,
        {reduce_name}{{eps}},
        {transform_name}{{weight.data()}});
  }}
}};

}}  // namespace infini::ops

#endif
"""


def _gen_reduce_transform_cpu(
    op: InfiniOpDef,
    dag: ComputeDAG,
    match: MatchResult,
    guard: str,
    op_snake: str,
) -> str:
    prefix = _dsl_prefix(op)
    reduce_op = _generate_reduce_op_cpu(op, dag, match)
    transform_op = _generate_transform_op_cpu(op, dag, match)
    base_header = f"base/{op_snake}.h"
    reduce_name = f"{prefix}Cpu{op.name}Reduce"
    transform_name = f"{prefix}Cpu{op.name}Transform"
    impl_suffix = ", Impl::kDsl" if op.impl_index > 0 else ""
    impl_include = (
        f'#include "impl.h"\n#include "cpu/{op_snake}/registry.h"\n'
        if op.impl_index > 0
        else ""
    )

    type_list = "ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>"

    return f"""\
#ifndef {guard}
#define {guard}

#include "cpu/templates/reduce_transform.h"
#include "{base_header}"
{impl_include}
namespace infini::ops {{

{reduce_op}

{transform_op}

template <>
class Operator<{op.name}, Device::Type::kCpu{impl_suffix}> : public {op.name} {{
 public:
  using {op.name}::{op.name};

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {{
    CpuReduceThenTransform<{type_list}>(
        input, out, batch_size_, nhead_, dim_,
        out.dtype(), input_strides_, out_strides_,
        {reduce_name}{{eps}},
        {transform_name}{{weight.data()}});
  }}
}};

}}  // namespace infini::ops

#endif
"""
