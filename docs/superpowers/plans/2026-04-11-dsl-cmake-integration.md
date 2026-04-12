# DSL Compiler CMake Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify code generation so `python -m dsl` replaces `generate_wrappers.py` as the single CMake entry point for all generated code (DSL kernels, pybind11 bindings, C API).

**Architecture:** Move libclang-based binding generation from `scripts/generate_wrappers.py` into `dsl/compiler/bindings.py`. The DSL `__main__.py` calls it after DSL generation. CMake invokes `python -m dsl` instead of `generate_wrappers.py`. The old script is retained as fallback.

**Tech Stack:** Python (DSL compiler), libclang (C++ parsing), pybind11 (bindings), CMake.

**Spec:** `docs/superpowers/specs/2026-04-11-dsl-cmake-integration-design.md`

---

## Task 1: Extract binding generation into `dsl/compiler/bindings.py`

**Files:**
- Create: `dsl/compiler/bindings.py`

- [ ] **Step 1: Create `dsl/compiler/bindings.py`**

Move the following from `scripts/generate_wrappers.py` into this new module:

1. **`_OperatorExtractor` class** (lines 27-90) — libclang AST parsing of `src/base/*.h`. Keep it as-is.

2. **`_find_optional_tensor_params()`** and **`_find_vector_tensor_params()`** (lines 95-112) — regex-based parameter detection.

3. **`_generate_pybind11()`** (lines 115-250) — pybind11 binding code generation, including per-op `impl_names` string overloads.

4. **`_generate_legacy_c()`** (lines 253-464) — C API source/header generation.

5. **`_snake_to_pascal()`** and **`_get_all_ops()`** (lines 467-489) — utility functions.

Wrap everything in a single entry point:

```python
def generate_all_bindings(
    devices: list[str],
    output_dir: pathlib.Path,
    impl_names: dict[str, dict[str, int]],
) -> None:
    """Generate pybind11 bindings and C API for all operators.

    This replaces the standalone `scripts/generate_wrappers.py` script.
    The libclang parsing, pybind11 generation, and C API generation
    logic is moved here verbatim.
    """
```

This function should:
1. Discover all ops via `_get_all_ops(devices)` (or `ops.json` if it exists).
2. For each op: parse with `_OperatorExtractor`, generate pybind11 binding header, generate C API files.
3. Assemble `ops.cc` with all includes and `PYBIND11_MODULE`.

Keep the same output paths: `generated/bindings/`, `generated/include/`, `generated/src/`.

Constants to define at module level:
```python
_SRC_DIR = pathlib.Path("src")
_BASE_DIR = _SRC_DIR / "base"
_INDENTATION = "  "
```

**Important:** This is a move, not a rewrite. Copy the functions verbatim from `generate_wrappers.py`, only adjusting imports and making them module-level instead of `if __name__ == "__main__"` scoped.

- [ ] **Step 2: Verify the module imports cleanly**

Run: `python -c "from dsl.compiler.bindings import generate_all_bindings; print('OK')"`
Expected: "OK"

- [ ] **Step 3: Commit**

```
git add dsl/compiler/bindings.py
git commit -m "refactor(dsl): extract binding generation into dsl/compiler/bindings.py"
```

---

## Task 2: Wire bindings into `dsl/__main__.py`

**Files:**
- Modify: `dsl/__main__.py`

- [ ] **Step 1: Add binding generation call**

At the end of `main()`, after the `impl_names.json` write and before the verify/summary print, add:

```python
if not args.verify:
    from dsl.compiler.bindings import generate_all_bindings
    generate_all_bindings(args.devices, args.output, all_impl_names)
```

Note: `all_impl_names` is already computed by `REGISTRY.all_impl_names()` earlier in `main()`. But the binding generator needs the full set (all ops, not just `--ops` filtered). The current `all_impl_names` call already covers all registered ops.

**Important detail:** The `generate_all_bindings` function discovers ops by scanning `src/base/*.h` (via `_get_all_ops`), independently of the DSL registry. This is correct — it needs to generate bindings for ALL operators, including `@manual_op` ones that have no DSL variant.

The `devices` list passed to binding generation must include `"cpu"` if `WITH_CPU` is enabled. Check that `args.devices` includes CPU. The existing `generate_wrappers.py` receives `${DEVICE_LIST}` from CMake which includes `cpu` when `WITH_CPU=ON`.

- [ ] **Step 2: Test the unified pipeline**

```bash
python -m dsl --devices cpu nvidia --output generated
```

Expected: generates all DSL kernel files + bindings + C API + impl_names.json.

Verify output matches `generate_wrappers.py`:
```bash
# Save current generated output.
cp -r generated /tmp/dsl_generated

# Run old script.
python scripts/generate_wrappers.py --devices cpu nvidia

# Compare bindings (the part that matters).
diff generated/bindings/ops.cc /tmp/dsl_generated/bindings/ops.cc
```

The outputs should be identical (or differ only in include ordering, which is harmless).

- [ ] **Step 3: Commit**

```
git add dsl/__main__.py
git commit -m "feat(dsl): integrate binding generation into python -m dsl"
```

---

## Task 3: Update CMakeLists.txt

**Files:**
- Modify: `src/CMakeLists.txt`

- [ ] **Step 1: Replace `generate_wrappers.py` with `python -m dsl`**

Change the `execute_process` call (around line 229-233):

From:
```cmake
execute_process(
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/generate_wrappers.py --devices ${DEVICE_LIST}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE script_result
)
```

To:
```cmake
execute_process(
    COMMAND ${Python_EXECUTABLE} -m dsl --devices ${DEVICE_LIST}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    RESULT_VARIABLE script_result
)
```

Also update the status message:
```cmake
if(NOT script_result EQUAL 0)
    message(FATAL_ERROR "DSL compilation and binding generation - failed")
else()
    message(STATUS "DSL compilation and binding generation - done")
endif()
```

- [ ] **Step 2: Build and verify**

```bash
pip install -e .[dev]
```

Expected: builds successfully using `python -m dsl` instead of `generate_wrappers.py`.

- [ ] **Step 3: Smoke test**

```bash
python -c "
import torch, infini.ops
a = torch.randn(4, 4, device='cuda')
b = torch.randn(4, 4, device='cuda')
out = torch.empty(4, 4, device='cuda')
infini.ops.add(a, b, out, implementation='dsl')
print('OK')
"
```

- [ ] **Step 4: Commit**

```
git add src/CMakeLists.txt
git commit -m "build: replace generate_wrappers.py with python -m dsl in CMake"
```

---

## Task 4: Full regression test

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ dsl/tests/ --tb=short -q \
    --ignore=tests/test_add_rms_norm.py \
    --ignore=tests/test_cat.py \
    --ignore=tests/test_linear.py \
    --ignore=tests/test_matmul.py
```

Expected: 4372+ passed, 0 failed.

- [ ] **Step 2: Run linter**

```bash
ruff check dsl/compiler/bindings.py dsl/__main__.py
ruff format dsl/compiler/bindings.py dsl/__main__.py
```

- [ ] **Step 3: Commit any lint fixes**

```
git add -u && git commit -m "style: fix lint issues"
```
