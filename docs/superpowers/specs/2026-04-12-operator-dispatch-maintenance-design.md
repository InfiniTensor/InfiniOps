# Operator Dispatch and Maintenance Optimization

## Problem

As operators and platforms grow, maintenance cost scales as `O(ops × platforms)`.
Each new platform requires a wrapper file per operator; each new operator
requires a wrapper per platform.  Currently, DSL-generated wrappers are
copied manually into `src/`, and `src/<platform>/` mixes adapter files
(4 per platform) with per-operator wrappers (1 per operator).

## Goal

Reduce the per-operator-per-platform cost to zero for CUDA-like platforms.
New platform onboarding: provide 4 adapter files, add a CMake flag, build.
New operator onboarding: write base class + CUDA kernel + DSL registration,
build.  All wrappers generated automatically.

## Design

### Directory responsibility separation

**`src/` — hand-written code only**

```
src/
  base/<op>.h                   # Abstract base class
  cuda/<op>/kernel.cuh          # Shared CUDA kernel
  cuda/<op>/kernel.h            # Shared CUDA launcher (CudaOp<Backend>)
  cuda/templates/               # Reusable brick templates
  cpu/<op>/<op>.h               # CPU implementation
  nvidia/                       # Platform adapter files ONLY:
    device_.h
    runtime_.h
    data_type_.h
    caster.cuh
    blas.h
    blas_utils.h
  metax/                        # Same 4-6 adapter files
    device_.h, runtime_.h, ...
  iluvatar/                     # Same
  moore/                        # Same
  ascend/                       # Ascend-specific impls (aclnn, not CUDA-like)
    <op>/kernel.h
  cambricon/                    # Cambricon-specific impls
    <op>/<op>.h
```

No per-operator wrapper files in `src/nvidia/`, `src/metax/`, etc.

**`generated/` — all auto-generated code**

```
generated/
  nvidia/<op>/kernel.h          # Operator<Op, kNvidia> wrapper
  metax/<op>/kernel.h           # Operator<Op, kMetax> wrapper
  iluvatar/<op>/kernel.h        # ...
  moore/<op>/kernel.h
  cpu/<op>/dsl.h                # DSL CPU impl (if @infini_op)
  nvidia/<op>/dsl.h             # DSL CUDA impl (if @infini_op)
  nvidia/<op>/registry.h        # ActiveImplementationsImpl (if multi-impl)
  cpu/<op>/registry.h           # ...
  bindings/*.h                  # pybind11 bindings
  bindings/ops.cc               # PYBIND11_MODULE
  include/*.h                   # C API headers
  src/*/operator.cc             # C API sources
  impl_names.json               # Per-op implementation name mapping
```

### CMake changes

Add `generated/<platform>/` to the source GLOB for each CUDA-like backend:

```cmake
if(WITH_NVIDIA)
    set(NVIDIA_PATTERNS
        "cuda/*.cc" "cuda/*.cpp" "cuda/*.cu"
        "nvidia/*.cc" "nvidia/*.cpp" "nvidia/*.cu"
    )
    file(GLOB_RECURSE NVIDIA_SOURCES CONFIGURE_DEPENDS ${NVIDIA_PATTERNS})

    # Add DSL-generated wrappers.
    file(GLOB_RECURSE NVIDIA_GENERATED CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/generated/nvidia/*.h"
    )

    # ... (wrapper .h files are header-only, included by ops.cc)
endif()
```

Since wrapper files are headers (not `.cc`), they are pulled in via
`#include` from the generated `ops.cc`.  The CMake change is mainly about
ensuring the include path covers `generated/`.

### DSL compiler changes

`python -m dsl --devices ${DEVICE_LIST}` already generates:
- `@infini_op` kernel files (cuda/cpu DSL code)
- Backend wrappers for CUDA-like platforms
- Bindings, C API, impl_names.json

**Changes needed:**
1. Generate `@manual_op` wrappers to `generated/` instead of relying on
   `generate_wrappers.py` scanning `src/`.
2. Remove the `_get_all_ops(devices)` scan-based discovery.  All ops are
   already registered in `dsl/ops/*.py` — use the registry directly.
3. The generated `ops.cc` includes should reference `generated/<platform>/`
   paths instead of `src/<platform>/`.

### New platform onboarding flow

```
1.  mkdir src/<platform>/
2.  Create: device_.h, runtime_.h, data_type_.h, caster.cuh
3.  CMakeLists.txt: add WITH_<PLATFORM> option, GLOB patterns, link libs
4.  pip install -e .[dev]   ← DSL auto-generates all wrappers
```

No operator-specific files needed.  The DSL compiler reads the `--devices`
list and generates `Operator<Op, kPlatform>` wrappers for every registered
operator.

### New operator onboarding flow

```
1.  Create src/base/<op>.h          (base class)
2.  Create src/cuda/<op>/kernel.cuh (CUDA kernel)
3.  Create src/cuda/<op>/kernel.h   (CUDA launcher: CudaOp<Backend>)
4.  Create dsl/ops/<op>.py          (@manual_op or @infini_op)
5.  Create tests/test_<op>.py       (tests)
6.  pip install -e .[dev]           ← wrappers + bindings auto-generated
```

For Ascend/Cambricon (non-CUDA-like): also add `src/ascend/<op>/kernel.h`
and reference it in `manual_backends` of the DSL definition.

### Migration plan

1. Move existing `src/nvidia/<op>/kernel.h` wrappers to `generated/`.
2. Move existing `src/nvidia/<op>/dsl.h` to `generated/`.
3. Move existing `src/nvidia/<op>/registry.h` to `generated/`.
4. Same for cpu DSL files and registries.
5. Keep `src/nvidia/` with only adapter files.
6. Update `ops.cc` includes from `src/nvidia/<op>/` to
   `generated/nvidia/<op>/`.
7. Verify full test suite passes.

### What stays unchanged

- `src/base/` — base classes (hand-written)
- `src/cuda/` — shared CUDA kernels and templates (hand-written)
- `src/cpu/` — hand-written CPU implementations
- `src/ascend/`, `src/cambricon/` — vendor-API implementations (hand-written)
- `src/operator.h`, `src/dispatcher.h` — core framework
- DSL decorator format (`@manual_op` / `@infini_op`)
- Python test framework

## Verification

```bash
pip install -e .[dev]
pytest tests/ dsl/tests/ --tb=short -q \
    --ignore=tests/test_add_rms_norm.py \
    --ignore=tests/test_cast.py \
    --ignore=tests/test_cat.py \
    --ignore=tests/test_linear.py \
    --ignore=tests/test_matmul.py
```

All tests must pass with zero wrapper files in `src/nvidia/*/`.

## References

- [PyTorch native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)
- [PyTorch Operator Registration](https://docs.pytorch.org/docs/stable/accelerator/operators.html)
- [ATen native README](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md)
