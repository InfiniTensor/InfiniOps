# Host Range Profiling Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and remotely validate an opt-in, host-only nested range collector for the existing InfiniOps pytest benchmark path.

**Architecture:** A compile-gated C++ RAII scope records fixed layer identifiers into a thread-local collector. Private pybind controls delimit cold and warm pytest windows, while a Python writer expands C++ inclusive/self summaries into JSON-lines records with separate `unit`, `mean`, and `median` fields.

**Tech Stack:** C++17, `std::chrono::steady_clock`, pybind11, pytest, `torch.utils.benchmark`, CMake, NVIDIA CUDA build on `ssh nvidia`.

---

### Task 1: Define collector behavior with Python-facing tests

**Files:**
- Create: `tests/test_host_range_profile.py`
- Modify: `tests/test_generate_wrappers.py`

- [ ] Add a generator test asserting that generated bindings include private
  start/stop/calibration controls and wrap free operator calls in
  `HostRangeLayer::kBindingBody`.
- [ ] Add profiling-build tests that skip when `_host_range_profile_compiled()`
  is false and otherwise verify start/stop state errors, nested inclusive/self
  ordering, stable layer names, and calibration counts.
- [ ] Run `pytest tests/test_generate_wrappers.py tests/test_host_range_profile.py -q`
  before implementation and confirm the new assertions fail.

### Task 2: Implement the compile-gated C++ collector

**Files:**
- Create: `src/host_range_profiler.h`
- Create: `src/host_range_profiler.cc`
- Modify: `CMakeLists.txt`

- [ ] Add `INFINI_OPS_ENABLE_HOST_RANGE_PROFILING`, default `OFF`, and publish
  the matching compile definition from the `infiniops` target.
- [ ] Define fixed layers for binding body, tensor/device conversion, generated
  dispatch, operator call, cache key, cache lookup, cache construction,
  operator invocation, backend submission, and three calibration depths.
- [ ] Implement `HostRangeScope` with a thread-local stack. On scope exit,
  append inclusive and self nanoseconds to the layer's sample vectors and add
  inclusive time to the direct parent's child accumulator.
- [ ] Implement strict `Start`, `Stop`, and calibration APIs. `Stop` sorts copies
  of the sample vectors and returns exact count, arithmetic mean, and median.
- [ ] Make `INFINI_OPS_HOST_RANGE_SCOPE(layer)` expand to no code when the CMake
  option is disabled.

### Task 3: Expose private controls and instrument generated boundaries

**Files:**
- Modify: `scripts/generate_wrappers.py`
- Modify: `src/pybind11_utils.h`
- Modify: `tests/test_generate_wrappers.py`

- [ ] Generate private pybind controls in `generated/bindings/ops.cc`. Convert
  each C++ summary to dictionaries containing `range`, `count`, `unit`,
  `inclusive_mean`, `inclusive_median`, `self_mean`, and `self_median`.
- [ ] Add `binding.body` to generated free-call lambdas and `dispatch.call` to
  generated `Call<Op>` functions.
- [ ] Add `binding.tensor_conversion` to `TensorFromPybind11Handle`.
- [ ] Run the focused generator tests and confirm generated code contains the
  scopes exactly once per boundary.

### Task 4: Instrument cache and selected backend host submission

**Files:**
- Modify: `src/operator.h`
- Modify: `src/native/cuda/ops/add/kernel.h`
- Modify: `src/native/cuda/nvidia/ops/gemm/cublaslt.h`

- [ ] Add nested scopes around the complete operator call, cache-key build,
  cache lookup, miss-only construction, and invocation.
- [ ] Add `backend.submit` around the Add launch and cuBLASLt host setup/call.
  End at API return and do not add stream/device synchronization.
- [ ] Run formatting checks on all changed C++ and Python files.

### Task 5: Integrate cold/warm collection with pytest

**Files:**
- Create: `tests/host_range_profile.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_host_range_profile.py`

- [ ] Add `--host-range-profile PATH`, valid only with `--benchmark` and a
  profiling-enabled extension.
- [ ] Before the InfiniOps benchmark, clear the selected operator cache, collect
  one cold call, then collect the warm `blocked_autorange()` window. Stop before
  running the reference timer.
- [ ] Expand each C++ range summary to separate inclusive/self JSON-lines rows.
  Attach nodeid, operator, backend, phase, count, unit, mean, and median.
- [ ] Record the `torch.utils.benchmark` end-to-end mean/median as a separate
  `end_to_end` range and leave the pre-lambda residual unattributed.
- [ ] Add unit tests for row expansion, output truncation at session start, and
  the compile-disabled usage error.

### Task 6: Build and validate on NVIDIA

**Files:**
- Create outside the repository: remote build and result directories under a
  unique `/tmp/infiniops-host-range-*` directory.

- [ ] Transfer the exact branch state to `ssh nvidia` and build InfiniRT plus
  InfiniOps in `accelerator-dev/nvidia:latest` with
  `CMAKE_BUILD_TYPE=RelWithDebInfo`, profiling enabled, and only Add/GEMM.
- [ ] Run the profiling-build unit tests and the two exact selectors:

  ```bash
  python3 -m pytest -q --devices nvidia --benchmark \
    --host-range-profile reports/add.jsonl \
    'tests/test_add.py::test_add[cuda-0-dtype0-1e-07-1e-07-shape0-None-None-None]'

  python3 -m pytest -q --devices nvidia --benchmark \
    --host-range-profile reports/gemm.jsonl \
    'tests/test_gemm.py::test_gemm[cuda-1-dtype0-0.001-0.001-False-False-0-1-a_shape4-b_shape4-c_shape4-None-None-None]'
  ```

- [ ] Build the same commit with profiling disabled and rerun both selectors as
  the end-to-end perturbation baseline.
- [ ] Run empty-window and three-level empty-range calibration and calculate the
  observer cost relative to each reported warm layer.

### Task 7: Review and package the experiment

**Files:**
- Create outside the repository: compact JSON, command log, environment dump,
  and Markdown interpretation under the task visualization directory.

- [ ] Verify both commits, compiler flags, image, CUDA/driver/GPU versions, exact
  selectors, test outcomes, and generated report sizes.
- [ ] Check that cold data contains `cache.construct`, warm data does not, and
  no measured code added a synchronize call.
- [ ] Run `git diff --check`, focused pytest tests, and the repository's format
  checks; record any validation unavailable locally.
- [ ] Report the actual inclusive/self statistics, calibration overhead, known
  attribution gaps, and whether pure ranges are suitable for the next phase.
