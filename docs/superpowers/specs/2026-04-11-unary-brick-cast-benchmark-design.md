# Unary Elementwise Brick, Cast Migration, and DSL Performance Benchmark

## Problem

The DSL currently has two brick templates (`binary_elementwise` and
`reduce_transform`) covering two-input elementwise and reduction-based
operators. Single-input operators like Cast cannot be expressed.
Additionally, there is no systematic performance comparison between
DSL-generated and hand-written kernel code.

## Solution

1. Add `UnaryElementwiseBrick` template (CUDA + CPU).
2. Migrate Cast to `@infini_op` using the new brick.
3. Benchmark all DSL-migrated operators against hand-written versions.

---

## 1. `UnaryElementwiseBrick`

### CUDA template (`src/cuda/templates/unary_elementwise.cuh`)

A single-input elementwise kernel with dual-dtype dispatch.

```cpp
template <Device::Type kDev, typename Op, typename TIn, typename TOut,
          unsigned int BLOCK_SIZE>
__global__ void UnaryElementwiseKernel(
    TOut* __restrict__ out, const TIn* __restrict__ in,
    const size_t* __restrict__ out_shape,
    const size_t* __restrict__ in_shape,
    const ptrdiff_t* __restrict__ out_strides,
    const ptrdiff_t* __restrict__ in_strides,
    size_t output_size, size_t ndim,
    bool out_contig, bool in_contig);
```

**Key differences from `BinaryElementwiseBrick`:**
- Single input tensor (no `other`).
- Dual-dtype dispatch: `DispatchFunc<InputTypes, OutputTypes>` resolves
  `(TIn, TOut)` at runtime from `(input_dtype, output_dtype)`.
- Op functor signature: `TOut operator()(const TIn& x) const`.

**`UnaryElementwiseBrick<Backend>` class:**
- Constructor takes `(input, out, ndim)` — allocates device metadata for
  two tensors (not three).
- `Run<InputTypeList, OutputTypeList, Op>()` does the dual dispatch and
  kernel launch.

### CPU template (`src/cpu/templates/unary_elementwise.h`)

```cpp
template <typename InputTypeList, typename OutputTypeList, typename Op>
void CpuUnaryElementwise(
    const Tensor in, Tensor out, Tensor::Size output_size,
    Tensor::Size ndim, bool in_contig, bool out_contig,
    const Tensor::Shape& in_shape, const Tensor::Shape& out_shape,
    const Tensor::Strides& in_strides, const Tensor::Strides& out_strides,
    DataType input_dtype, DataType output_dtype, Op op);
```

Uses `DispatchFunc` with two `DataType` lists for dual dispatch, OpenMP
parallel for loop, and `Caster` for type conversion.

### Future reuse

Although Cast is the immediate use case, the unary brick also serves future
single-input operators (ReLU, GELU, Sigmoid, Abs, Neg). Those have
`input_dtype == output_dtype`, which works naturally — dual dispatch
resolves both to the same type.

---

## 2. Cast DSL migration

### DSL definition

```python
# dsl/ops/cast_dsl.py
@infini_op(name="Cast", impl_index=1, shapes={"N": "output_size"})
def cast_dsl(input: Tensor["N"]) -> Tensor["N"]:
    return cast(input)
```

### Compiler changes

**`dsl/compiler/patterns.py`:**
- Add `BrickKind.UNARY_ELEMENTWISE`.
- Match rule: single input, no reduction, single output → unary.

**`dsl/compiler/infini_codegen.py`:**
- Add `_gen_unary_elementwise_cuda()` and `_gen_unary_elementwise_cpu()`.
- Cast functor body: `Caster<kDev>::Cast<TOut>(x)` (pure type conversion,
  no math).
- Generated class `DslCudaCast` inherits from `Cast` base class.

**`dsl/__main__.py`:**
- Route `UNARY_ELEMENTWISE` brick to the new generators.
- Output paths: `cuda/cast/dsl.h`, `nvidia/cast/dsl.h`, `cpu/cast/dsl.h`,
  plus `registry.h` files.

### Registration

- `Operator<Cast, kNvidia, Impl::kDsl>` via generated nvidia wrapper.
- `Operator<Cast, kCpu, Impl::kDsl>` via generated CPU file.
- `registry.h` files for nvidia and CPU.
- Cast currently has no NVIDIA hand-written implementation, so the nvidia
  registry declares `List<Impl::kDsl>` only (dispatcher fallback handles
  default index).

---

## 3. Performance benchmark

### Test file

`tests/benchmark_dsl.py`, using `@pytest.mark.benchmark` (only runs with
`pytest --benchmark`).

### Test matrix

| Operator | Shapes | Dtypes | Compare |
|----------|--------|--------|---------|
| Add | (4,4,5632), (16,5632), (1024,1024) | fp32, fp16, bf16 | default vs dsl |
| RmsNorm | (2,4,2048), (4,48,64) | fp32, fp16, bf16 | default vs dsl |
| Swiglu | (4,4,5632), (16,5632) | fp32, fp16, bf16 | default vs dsl |
| Cast | (4,4,5632), (1024,1024) | fp32→fp16, fp16→fp32 | default vs dsl |

Mul is excluded (NVIDIA has DSL-only, no hand-written to compare).

### Measurement

- CUDA event timing (`torch.cuda.Event`) for GPU kernel time.
- Warmup runs + multiple iterations, report median.
- Output: table with `hand-written ms`, `dsl ms`, `ratio`.

### Success criterion

DSL-generated code within 80-120% of hand-written performance (per the
design spec's 10-20% tolerance target).

---

## Files to create/modify

| File | Action |
|------|--------|
| `src/cuda/templates/unary_elementwise.cuh` | New: CUDA unary brick |
| `src/cpu/templates/unary_elementwise.h` | New: CPU unary brick |
| `dsl/compiler/patterns.py` | Modify: add `UNARY_ELEMENTWISE` |
| `dsl/compiler/infini_codegen.py` | Modify: add unary codegen |
| `dsl/__main__.py` | Modify: route unary brick |
| `dsl/ops/cast_dsl.py` | New: Cast DSL definition |
| `src/cuda/cast/dsl.h` | New: generated CUDA kernel |
| `src/nvidia/cast/dsl.h` | New: generated nvidia wrapper |
| `src/cpu/cast/dsl.h` | New: generated CPU impl |
| `src/{nvidia,cpu}/cast/registry.h` | New: impl registry |
| `src/cpu/cast/cast.h` | Modify: add registry include |
| `tests/benchmark_dsl.py` | New: performance benchmark |

## Verification

```bash
pip install -e .[dev]
pytest tests/test_cast.py -v                    # existing Cast tests
pytest tests/test_cast_dsl.py -v                # new DSL Cast tests
pytest tests/ --ignore=... --tb=short           # full regression
pytest tests/benchmark_dsl.py --benchmark -v    # performance comparison
```
