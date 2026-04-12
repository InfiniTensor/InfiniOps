# DSL Compiler CMake Integration

## Problem

The build system runs `generate_wrappers.py` for pybind11 bindings and C
API generation, while `python -m dsl` is a separate manual step for DSL
kernel generation.  This dual-system setup means:

- DSL-generated files must be pre-generated before `pip install`.
- `impl_names.json` must exist before `generate_wrappers.py` runs.
- New operators require touching both systems.

## Solution

Unify code generation into `python -m dsl`, which absorbs all functionality
from `generate_wrappers.py`.  CMake calls one command.
`generate_wrappers.py` is retained as a fallback but not called by CMake.

---

## Architecture

### Before

```
CMakeLists.txt
  └─ execute_process(generate_wrappers.py --devices ...)
       ├─ libclang parse src/base/*.h
       ├─ scan src/ for Operator<> specializations
       └─ emit: generated/bindings/*.h, ops.cc, include/*.h, src/*/operator.cc

Manual step:
  └─ python -m dsl --output generated --devices ...
       ├─ emit: DSL kernel files (cuda/*/dsl.h, etc.)
       ├─ emit: registry.h files
       └─ emit: impl_names.json
```

### After

```
CMakeLists.txt
  └─ execute_process(python -m dsl --devices ...)
       ├─ DSL kernel generation (unchanged)
       ├─ registry.h generation (unchanged)
       ├─ impl_names.json generation (unchanged)
       ├─ libclang parse src/base/*.h (moved from generate_wrappers.py)
       ├─ scan src/ for Operator<> specializations (moved)
       └─ emit: generated/bindings/*.h, ops.cc, include/*.h, src/*/operator.cc
```

`generate_wrappers.py` remains in `scripts/` as a fallback.  It is not
called by CMake.  It can be used to verify output consistency during the
transition period.

---

## Implementation

### 1. Create `dsl/compiler/bindings.py`

Move from `generate_wrappers.py`:
- `_OperatorExtractor` class (libclang AST parsing)
- `_generate_pybind11()` function (pybind11 binding generation)
- `_generate_legacy_c()` function (C API generation)
- Helper functions: `_find_optional_tensor_params()`,
  `_find_vector_tensor_params()`, `_snake_to_pascal()`

The module exposes one entry point:

```python
def generate_all_bindings(
    devices: list[str],
    output_dir: pathlib.Path,
    impl_names: dict[str, dict[str, int]],
) -> None:
```

This function:
1. Discovers all operators via `src/base/*.h` (same logic as
   `_get_all_ops()` in `generate_wrappers.py`).
2. For each operator, parses the base class with libclang, generates
   pybind11 bindings (with per-op `impl_names` string overloads) and
   C API files.
3. Assembles `ops.cc` with all includes and `PYBIND11_MODULE`.

### 2. Update `dsl/__main__.py`

After the existing DSL generation loop, call:

```python
from dsl.compiler.bindings import generate_all_bindings
generate_all_bindings(args.devices, args.output, all_impl_names)
```

This replaces the separate `generate_wrappers.py` invocation.

### 3. Update `src/CMakeLists.txt`

Replace:
```cmake
execute_process(
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/generate_wrappers.py
            --devices ${DEVICE_LIST}
    ...
)
```

With:
```cmake
execute_process(
    COMMAND ${Python_EXECUTABLE} -m dsl --devices ${DEVICE_LIST}
    ...
)
```

### 4. Keep `generate_wrappers.py` as fallback

No changes to `scripts/generate_wrappers.py`.  It can be run manually to
verify output consistency:

```bash
# Compare outputs.
python -m dsl --devices nvidia --output /tmp/dsl_out
python scripts/generate_wrappers.py --devices nvidia
diff -r generated/ /tmp/dsl_out/
```

---

## Files to create/modify

| File | Action |
|------|--------|
| `dsl/compiler/bindings.py` | New: libclang parsing + binding generation (moved from generate_wrappers.py) |
| `dsl/__main__.py` | Modify: call `generate_all_bindings()` after DSL generation |
| `src/CMakeLists.txt` | Modify: replace `generate_wrappers.py` with `python -m dsl` |

## What stays unchanged

- `scripts/generate_wrappers.py` — retained as fallback, not called by CMake
- All existing DSL generation logic in `dsl/compiler/`
- libclang parsing logic (moved, not rewritten)
- Generated output format (bindings, C API, ops.cc)

## Verification

```bash
# Build with unified pipeline.
pip install -e .[dev]

# Verify bindings work.
python -c "import infini.ops; print(dir(infini.ops))"

# Verify string implementation param works.
python -c "
import torch, infini.ops
a = torch.randn(4, 4, device='cuda')
b = torch.randn(4, 4, device='cuda')
out = torch.empty(4, 4, device='cuda')
infini.ops.add(a, b, out, implementation='dsl')
print('OK')
"

# Full test suite.
pytest tests/ dsl/tests/ --tb=short -q \
    --ignore=tests/test_add_rms_norm.py \
    --ignore=tests/test_cat.py \
    --ignore=tests/test_linear.py \
    --ignore=tests/test_matmul.py

# Compare with legacy script output (optional).
python scripts/generate_wrappers.py --devices cpu nvidia
diff generated/bindings/ops.cc /tmp/legacy_ops.cc
```
