# InfiniOps Cross-Platform DSL Design

## Problem

Adding a new operator to InfiniOps requires 10+ files and ~670 lines of code,
roughly 50% of which is boilerplate. CUDA-like backends (NVIDIA, MetaX,
Iluvatar, Moore) share ~99% of kernel code via `src/cuda/` templates, yet each
still needs a hand-written 21-line wrapper file per operator. CPU
implementations duplicate the same mathematical logic in a separate
OpenMP-based form. The core algorithmic intent is expressed repeatedly across
backends rather than once.

Ascend uses aclnn vendor APIs exclusively and cannot share kernel code with
CUDA backends. Its implementations will remain hand-written.

## Solution

A Python DSL for defining operator semantics, paired with a C++ template
building-block library ("bricks"). The DSL compiler translates operator
definitions into C++ code that composes these bricks. Hand-written kernels
remain available for performance-critical or complex operators via an escape
hatch.

### Scope

- **Automated by DSL**: CUDA-like backends (NVIDIA, MetaX, Iluvatar, Moore) +
  CPU.
- **Hand-written, DSL-managed boilerplate**: Ascend (aclnn), Cambricon
  (cnnl/BANG), and any future vendor-API platform.
- **Performance target**: generated kernel code within 10-20% of hand-written.
  Performance-critical operators (GEMM, FlashAttention) use the escape hatch.

---

## 1. Python DSL

### Operator definition

Operators are Python functions decorated with `@infini_op`. The function body
uses a restricted set of tensor primitives to describe mathematical semantics
declaratively (no control flow, no side effects).

```python
# dsl/ops/rms_norm.py

from infini_dsl import infini_op, Tensor, Scalar, reduce_mean, rsqrt

@infini_op(
    name="RmsNorm",
    shapes={"B": "batch", "H": "heads", "D": "dim"},
)
def rms_norm(
    input: Tensor["B", "H", "D"],
    weight: Tensor["D"],
    eps: Scalar[float] = 1e-6,
) -> Tensor["B", "H", "D"]:
    ss = reduce_mean(input * input, dim="D")
    rms = rsqrt(ss + eps)
    return input * rms * weight
```

Shape variables (`B`, `H`, `D`) let the compiler infer grid/block mapping and
derive base-class member fields.

### Primitive set

| Category | Primitives |
|----------|------------|
| Elementwise | `+`, `-`, `*`, `/`, `sqrt`, `rsqrt`, `exp`, `log`, `abs`, `neg`, `pow`, `clamp` |
| Activation | `relu`, `gelu`, `silu`, `sigmoid`, `tanh` |
| Reduction | `reduce_sum`, `reduce_mean`, `reduce_max`, `reduce_min` |
| Softmax | `softmax`, `log_softmax` |
| Comparison | `where(cond, a, b)`, `>`, `<`, `>=`, `<=`, `eq` |
| Type | `cast(x, dtype)` |
| Shape | `reshape`, `transpose`, `unsqueeze`, `expand`, `cat`, `slice` |
| Index | `gather`, `scatter`, `index_select` |
| Scalar | `Scalar[float]`, `Scalar[int]` |

Operators that cannot be expressed with these primitives use `@manual_op`.

### Escape hatch

```python
@manual_op(
    name="Gemm",
    base="src/base/gemm.h",
    backends={
        "cuda": "src/cuda/gemm/blas.h",
        "ascend": "src/ascend/gemm/kernel.h",
        "cpu": "src/cpu/gemm/gemm.h",
    },
)
def gemm(): ...
```

`@manual_op` tells the compiler to generate only boilerplate (backend wrapper
files, Python bindings, test scaffolding) while leaving kernel logic to the
hand-written files specified in `backends`.

### Mixed mode

An `@infini_op` can specify `manual_backends` for platforms that need
hand-written implementations while still auto-generating for CUDA-like
backends and CPU:

```python
@infini_op(
    name="RmsNorm",
    manual_backends={
        "ascend": "src/ascend/rms_norm/kernel.h",
        "cambricon": "src/cambricon/rms_norm/rms_norm.h",
    },
)
def rms_norm(...):
    ...
```

CUDA-like backends and CPU get auto-generated code; Ascend and Cambricon use
the specified hand-written files. One decorator manages all backends.

---

## 2. DSL compiler

### Pipeline

```
Python DSL source → AST parse → Compute DAG → Pattern match → C++ codegen
```

**AST parse**: extracts the function signature (tensor shapes, dtypes, scalar
attributes) and body (primitive operations).

**Compute DAG**: a directed acyclic graph where nodes are primitive operations
and edges are tensor data flows. Shape variables propagate through the graph
for dimension inference.

**Pattern match**: the compiler maintains a set of pattern rules that map
subgraph shapes to template bricks:

```python
PATTERNS = [
    Pattern(match=all_elementwise,      emit="ElementwiseKernel"),
    Pattern(match=reduce_then_transform, emit="ReduceThenTransform"),
    Pattern(match=softmax_pattern,       emit="SoftmaxKernel"),
    Pattern(match=has_gather_scatter,    emit="IndexKernel"),
    Pattern(match=pure_reduction,        emit="ReductionKernel"),
]
```

If a subgraph cannot be matched, the compiler emits an error directing the
user to either decompose the operator or use `@manual_op`.

**C++ codegen**: emits C++ source files using Jinja2 templates. Generated code
calls template bricks with operator-specific functors.

### Directory structure

```
dsl/
  ops/                    # Operator definitions (@infini_op, @manual_op)
  compiler/
    __init__.py
    parser.py             # AST → compute DAG
    patterns.py           # Pattern matching rules
    codegen.py            # C++ code generation (CUDA-like + CPU)
  templates/              # Jinja2 templates for generated C++ files
    base_class.h.j2
    cuda_kernel.h.j2
    backend_wrapper.h.j2
    cpu_kernel.h.j2
    test.py.j2
```

### Invocation

```bash
python -m dsl.compiler --devices nvidia metax iluvatar moore cpu \
                        --output generated/
```

Integrated into CMake, runs before compilation. Replaces the current
`generate_wrappers.py` call (bindings generation is subsumed).

---

## 3. C++ template brick library

Hand-written, optimized C++ templates that serve as the code-generation
targets. Each brick is parameterized on `Device::Type kDev` and user-provided
functors, so the same brick serves all CUDA-like backends.

### Brick inventory

| Brick | Location | Covers |
|-------|----------|--------|
| `ElementwiseKernel` | `src/cuda/templates/elementwise.cuh` | Add, Mul, ReLU, GELU, SiLU, Sigmoid, Tanh, Cast, Abs, Neg |
| `BroadcastKernel` | `src/cuda/templates/broadcast.cuh` | Elementwise ops on different-shaped tensors |
| `ReductionKernel` | `src/cuda/templates/reduction.cuh` | ReduceSum, ReduceMean, ReduceMax, ReduceMin |
| `ReduceThenTransform` | `src/cuda/templates/reduce_transform.cuh` | RmsNorm, LayerNorm, L2Norm |
| `SoftmaxKernel` | `src/cuda/templates/softmax.cuh` | Softmax, LogSoftmax, CausalSoftmax |
| `IndexKernel` | `src/cuda/templates/index.cuh` | Gather, Scatter, IndexSelect, Embedding |
| `ShapeKernel` | `src/cuda/templates/shape.cuh` | Reshape, Transpose, Cat, Slice |

### Interface pattern

```cpp
// src/cuda/templates/elementwise.cuh

template <Device::Type kDev, typename F>
struct ElementwiseKernel {
    static void Run(
        typename Runtime<kDev>::Stream stream,
        const Tensor input,
        Tensor output,
        F op);
};
```

Bricks use `Caster<kDev>` for type conversions and `Runtime<kDev>` for memory
operations. This defers all platform-specific details to the existing
per-backend specializations.

### CPU counterparts

Each CUDA brick has a CPU counterpart in `src/cpu/templates/` using OpenMP:

```cpp
// src/cpu/templates/elementwise.h

template <typename F>
struct CpuElementwise {
    static void Run(const Tensor input, Tensor output, F op);
};
```

### Generated code example

For `rms_norm`, the compiler generates:

```cpp
// generated/cuda/rms_norm/kernel.h

template <typename Backend>
class CudaRmsNorm : public RmsNorm {
    void operator()(const Tensor input, const Tensor weight,
                    Tensor out) const override {
        ReduceThenTransform<Backend::kDeviceType>::Run(
            stream_, input, out,
            ReduceMeanSquare{},
            RsqrtEpsMulWeight{weight, eps_},
            dim_, batch_size_, nhead_);
    }
};
```

---

## 4. Generated output

### For `@infini_op` operators

```
generated/
  base/<op>.h                           # Abstract base class
  cuda/<op>/kernel.h                    # CudaOp<Backend> template (brick calls)
  nvidia/<op>/kernel.h                  # Operator<Op, kNvidia> wrapper
  metax/<op>/kernel.h                   # Operator<Op, kMetax> wrapper
  iluvatar/<op>/kernel.h               # Operator<Op, kIluvatar> wrapper
  moore/<op>/kernel.h                  # Operator<Op, kMoore> wrapper
  cpu/<op>/<op>.h                      # CPU implementation (OpenMP bricks)
  bindings/<op>.h                      # pybind11 bindings
  src/<op>/operator.cc                 # C API (legacy)
  tests/test_<op>.py                   # Parametrized tests
```

### For `@manual_op` operators

```
generated/
  nvidia/<op>/kernel.h                 # Wrapper pointing to hand-written cuda impl
  metax/<op>/kernel.h                  # Wrapper
  iluvatar/<op>/kernel.h              # Wrapper
  moore/<op>/kernel.h                 # Wrapper
  bindings/<op>.h                     # pybind11 bindings
  tests/test_<op>.py                  # Test scaffolding
```

Base class, kernel logic, and Ascend/Cambricon implementations remain in
`src/` under manual control.

### Unchanged files

- `src/cuda/templates/` — hand-written brick library.
- `src/ascend/` — all Ascend implementations.
- `src/operator.h`, `src/dispatcher.h`, `src/device.h` — core framework.
- `src/<backend>/runtime_.h`, `data_type_.h`, `caster.cuh` — platform
  adaptation layers.

---

## 5. New platform onboarding

### CUDA-compatible platforms

Provide four adaptation files:

```
src/<platform>/device_.h       # DeviceEnabled<kPlatform> = true
src/<platform>/runtime_.h      # Runtime<kPlatform>: Stream, Malloc, Free, Memcpy
src/<platform>/data_type_.h    # TypeMap specializations for fp16/bf16
src/<platform>/caster.cuh      # Type conversion specializations
```

Add `--devices <platform>` to the compiler invocation. All `@infini_op`
operators automatically get generated wrappers for the new platform. No
operator definitions need to change.

### Vendor-API platforms

Add the platform to `manual_backends` in each operator's `@infini_op` or
`@manual_op` definition:

```python
@infini_op(
    name="RmsNorm",
    manual_backends={
        "ascend": "src/ascend/rms_norm/kernel.h",
        "new_vendor": "src/new_vendor/rms_norm/kernel.h",
    },
)
```

Hand-write each operator implementation using the vendor's SDK. The compiler
generates wrappers and bindings.

---

## 6. Migration strategy

### Phase 1: `@manual_op` for all existing operators

Register every existing operator as `@manual_op`. This immediately eliminates
hand-written wrapper files (the ~21-line `Operator<Op, kBackend>` files) and
centralizes binding generation. No kernel code changes.

### Phase 2: Extract template bricks from existing kernels

Refactor existing hand-written CUDA kernels in `src/cuda/` into the template
brick library. The existing `CudaAdd`, `CudaRmsNorm`, etc. provide the
implementations.

### Phase 3: Migrate simple operators to `@infini_op`

Convert elementwise operators (Add, ReLU, Cast, SiLU, etc.) to DSL
definitions. Verify generated code matches existing behavior via tests.

### Phase 4: Migrate medium-complexity operators

Convert reduction-based operators (RmsNorm, LayerNorm, Softmax) to DSL
definitions using the `ReduceThenTransform` and `SoftmaxKernel` bricks.

### Non-migrated operators

GEMM, FlashAttention, RotaryEmbedding, and other complex/performance-critical
operators remain as `@manual_op` indefinitely. The DSL still manages their
boilerplate.

---

## 7. Verification

### Auto-generated tests

The compiler derives a PyTorch reference implementation directly from the DSL
function body and generates parametrized tests using the existing
`Payload`/`auto_act_and_assert` framework.

### Brick-level tests

```
tests/test_templates/
  test_elementwise.py
  test_reduction.py
  test_reduce_transform.py
  test_softmax.py
  test_index.py
```

### End-to-end

```bash
python -m dsl.compiler --devices nvidia metax iluvatar moore cpu \
                        --output generated/
pip install -e .[dev]
pytest tests/ -v --tb=short
pytest tests/ --devices ascend -v   # Ascend ops unaffected
```
