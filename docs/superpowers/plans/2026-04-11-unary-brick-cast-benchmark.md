# Unary Elementwise Brick, Cast Migration, and Performance Benchmark

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `UnaryElementwiseBrick` C++ template, migrate Cast to DSL, and benchmark all DSL operators against hand-written versions.

**Architecture:** New unary brick templates (CUDA + CPU) with dual-dtype dispatch handle single-input operators. The DSL compiler learns to match unary DAGs and emit code using these bricks. A benchmark script compares DSL vs hand-written kernel performance.

**Tech Stack:** C++17/CUDA (brick templates), Python (DSL compiler, benchmarks), pybind11 (bindings), pytest + `torch.utils.benchmark` (benchmarks).

**Spec:** `docs/superpowers/specs/2026-04-11-unary-brick-cast-benchmark-design.md`

---

## Task 1: CUDA unary elementwise brick

**Files:**
- Create: `src/cuda/templates/unary_elementwise.cuh`

- [ ] **Step 1: Create the CUDA unary kernel and brick class**

Model on `src/cuda/templates/binary_elementwise.cuh`. Key differences:
- One input tensor instead of two.
- Dual-dtype dispatch: `Run` takes `InputTypeList` and `OutputTypeList` and dispatches on `(input_dtype, output_dtype)`.
- Op functor signature: `TOut operator()(const TIn& x) const`.
- `UnaryElementwiseBrick<Backend>` manages device metadata for 2 tensors (input + output) instead of 3.

Use `DispatchFunc<InputTypeList, OutputTypeList>` with `{static_cast<int64_t>(input_dtype), static_cast<int64_t>(output_dtype)}` for mixed multi-type dispatch (see `CONTRIBUTING.md` "Mixed Multi-Type Dispatch" section). Inside the lambda, use `ListGet<0>(list_tag)` and `ListGet<1>(list_tag)` to extract both types.

- [ ] **Step 2: Verify it compiles**

Run: `pip install -e .[dev] 2>&1 | tail -3`
Expected: "Successfully installed InfiniOps-0.1.0"

- [ ] **Step 3: Commit**

```
git add src/cuda/templates/unary_elementwise.cuh
git commit -m "feat(dsl): add CUDA unary elementwise brick template"
```

---

## Task 2: CPU unary elementwise brick

**Files:**
- Create: `src/cpu/templates/unary_elementwise.h`

- [ ] **Step 1: Create the CPU unary elementwise function**

Model on `src/cpu/templates/binary_elementwise.h`. Key differences:
- Single input tensor.
- Dual-dtype dispatch: nested `DispatchFunc` calls — outer dispatches `input_dtype`, inner dispatches `output_dtype` (same pattern as existing `src/cpu/cast/cast.h`).
- Op functor signature: `TOut operator()(const TIn& x) const`.
- OpenMP parallel for loop with `IndexToOffset` for non-contiguous tensors.

- [ ] **Step 2: Verify it compiles**

Run: `pip install -e .[dev] 2>&1 | tail -3`
Expected: "Successfully installed InfiniOps-0.1.0"

- [ ] **Step 3: Commit**

```
git add src/cpu/templates/unary_elementwise.h
git commit -m "feat(dsl): add CPU unary elementwise brick template"
```

---

## Task 3: DSL compiler — unary codegen

**Files:**
- Modify: `dsl/compiler/infini_codegen.py` — add `_gen_unary_elementwise_cuda()`, `_gen_unary_elementwise_cpu()`, `_generate_unary_functor_cuda()`, `_generate_unary_functor_cpu()`
- Modify: `dsl/__main__.py` — route `BrickKind.UNARY_ELEMENTWISE` to new generators

Note: `dsl/compiler/patterns.py` already has `BrickKind.UNARY_ELEMENTWISE` and matching logic.

- [ ] **Step 1: Add unary functor generators to `infini_codegen.py`**

Add `_generate_unary_functor_cuda(op, dag, match)` and `_generate_unary_functor_cpu(op, dag, match)`. These follow the same pattern as `_generate_binary_functor_cuda/cpu` but with:
- Single input `va` instead of `va, vb`.
- Return type may differ from input type (for Cast).

For Cast specifically, the functor body is just `return Caster<kDev>::template Cast<TOut>(va);` (CUDA) or `return static_cast<TOut>(va);` (CPU).

- [ ] **Step 2: Add unary file generators to `infini_codegen.py`**

Add `_gen_unary_elementwise_cuda(op, dag, match, guard, op_snake)` and `_gen_unary_elementwise_cpu(...)`. These generate complete header files that:
- Include `cuda/templates/unary_elementwise.cuh` or `cpu/templates/unary_elementwise.h`.
- Include the base class header (`base/cast.h`).
- Define the functor struct and `DslCudaCast` / `Operator<Cast, kCpu, Impl::kDsl>` classes.
- Use `AllTypes` for both input and output type lists.
- The CUDA class constructor takes `(input, out)` matching Cast's base class.

- [ ] **Step 3: Wire `generate_cuda_kernel` and `generate_cpu_kernel` to handle `UNARY_ELEMENTWISE`**

Add `if match.brick == BrickKind.UNARY_ELEMENTWISE` branches in both functions.

- [ ] **Step 4: Update `__main__.py` to route unary brick**

In `_generate_infini_op`, the code already calls `generate_cuda_kernel` and `generate_cpu_kernel` which will now handle `UNARY_ELEMENTWISE`. No changes needed in `__main__.py` unless the output path logic differs. Verify by running:

```
python -m dsl --ops Cast --output /tmp/dsl_test --devices nvidia
```

Expected: generates `cuda/cast/dsl.h`, `cpu/cast/dsl.h`, `nvidia/cast/dsl.h`, registries.

- [ ] **Step 5: Commit**

```
git add dsl/compiler/infini_codegen.py dsl/__main__.py
git commit -m "feat(dsl): add unary elementwise codegen for @infini_op"
```

---

## Task 4: Cast DSL migration

**Files:**
- Create: `dsl/ops/cast_dsl.py`
- Create: `src/cuda/cast/dsl.h` (generated)
- Create: `src/nvidia/cast/dsl.h` (generated)
- Create: `src/cpu/cast/dsl.h` (generated)
- Create: `src/nvidia/cast/registry.h` (generated)
- Create: `src/cpu/cast/registry.h` (generated)
- Modify: `src/cpu/cast/cast.h` — add `#include "cpu/cast/registry.h"`
- Create: `tests/test_cast_dsl.py`

- [ ] **Step 1: Create DSL definition**

Create `dsl/ops/cast_dsl.py`:
```python
from dsl.decorators import infini_op
from dsl.primitives import Tensor, cast

@infini_op(
    name="Cast",
    impl_index=1,
    shapes={"N": "output_size"},
    manual_backends={
        "ascend": "ascend/cast/kernel.h",
    },
)
def cast_dsl(input: Tensor["N"]) -> Tensor["N"]:
    return cast(input)
```

- [ ] **Step 2: Generate and place files**

```
python -m dsl --ops Cast --output /tmp/dsl_cast --devices nvidia
```

Copy generated files to `src/`:
- `src/cuda/cast/dsl.h`
- `src/nvidia/cast/dsl.h`
- `src/cpu/cast/dsl.h`
- `src/cpu/cast/registry.h`

For nvidia, manually create `src/nvidia/cast/registry.h` with `List<Impl::kDsl>` only (no hand-written NVIDIA impl exists; dispatcher fallback handles default index).

- [ ] **Step 3: Update existing CPU cast to include registry**

Add `#include "cpu/cast/registry.h"` to `src/cpu/cast/cast.h`.

- [ ] **Step 4: Create test**

Create `tests/test_cast_dsl.py` following `tests/test_cast.py` pattern. Use `implementation="dsl"`. Test fp32→fp16, fp16→fp32, bf16→fp32, fp32→bf16 conversions.

- [ ] **Step 5: Regenerate `impl_names.json` and rebuild**

```
python -m dsl --output generated --devices nvidia
pip install -e .[dev]
```

- [ ] **Step 6: Run tests**

```
pytest tests/test_cast_dsl.py -v
pytest tests/test_cast.py --devices cpu -v   # existing tests (CPU only, no CUDA hand-written)
```

Expected: all pass.

- [ ] **Step 7: Commit**

```
git add dsl/ops/cast_dsl.py src/cuda/cast/dsl.h src/nvidia/cast/ src/cpu/cast/ tests/test_cast_dsl.py
git commit -m "feat(dsl): migrate Cast to @infini_op with unary elementwise brick"
```

---

## Task 5: Performance benchmark

**Files:**
- Create: `tests/benchmark_dsl.py`

- [ ] **Step 1: Create benchmark script**

Create `tests/benchmark_dsl.py` using `torch.utils.benchmark.Timer` and `@pytest.mark.benchmark`. Structure:

```python
import pytest
import torch
import torch.utils.benchmark as benchmark
import infini.ops

@pytest.mark.benchmark
@pytest.mark.parametrize("op_name, shape, dtype, setup_fn", [
    # Add
    ("add", (4, 4, 5632), torch.float32, _setup_binary),
    ("add", (1024, 1024), torch.float16, _setup_binary),
    # RmsNorm
    ("rms_norm", (2, 4, 2048), torch.float32, _setup_rms_norm),
    # Swiglu
    ("swiglu", (4, 4, 5632), torch.float32, _setup_binary),
    # Cast
    ("cast", (4, 4, 5632), torch.float32, _setup_cast),  # fp32→fp16
])
def test_benchmark_dsl_vs_default(op_name, shape, dtype, setup_fn):
    ...
```

Each test:
1. Creates tensors on CUDA.
2. Runs the operator with `implementation="default"` (hand-written) — times it.
3. Runs with `implementation="dsl"` — times it.
4. Computes ratio. Prints comparison table.
5. Asserts ratio is within 0.8-1.2 (configurable via marker).

Skip operators that lack a hand-written CUDA implementation (Mul, Cast on NVIDIA) — they only have DSL, so no comparison is possible.

- [ ] **Step 2: Run benchmark**

```
pytest tests/benchmark_dsl.py --benchmark -v --devices cuda
```

Expected: table of results showing DSL vs hand-written timing.

- [ ] **Step 3: Commit**

```
git add tests/benchmark_dsl.py
git commit -m "test(dsl): add performance benchmark comparing DSL vs hand-written kernels"
```

---

## Task 6: Full regression and final commit

- [ ] **Step 1: Run full test suite**

```
pytest tests/ dsl/tests/ --tb=short -q \
    --ignore=tests/test_add_rms_norm.py \
    --ignore=tests/test_cat.py \
    --ignore=tests/test_linear.py \
    --ignore=tests/test_matmul.py
```

Expected: 4300+ passed, 0 failed.

(The ignored tests are pre-existing CUDA crashes for operators without NVIDIA implementations — unrelated to this work.)

- [ ] **Step 2: Run linter**

```
ruff check dsl/ scripts/generate_wrappers.py tests/test_cast_dsl.py tests/benchmark_dsl.py
ruff format dsl/ tests/test_cast_dsl.py tests/benchmark_dsl.py
```
