# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

InfiniOps uses CMake + scikit-build-core. The library is compiled into a shared `libinfiniops` and an optional Python extension `ops`.

### C++ only

```bash
mkdir build && cd build
cmake .. -DWITH_CPU=ON           # or -DWITH_NVIDIA=ON, -DWITH_METAX=ON, etc.
make -j$(nproc)
```

### Python package (pip / editable install)

```bash
pip install .[dev]               # installs infiniops + dev tools
# or for an editable build:
pip install -e .[dev]
```

`pyproject.toml` sets `AUTO_DETECT_DEVICES=ON` and `GENERATE_PYTHON_BINDINGS=ON` automatically during `pip install`.

### Backend CMake flags

| Flag | Backend |
|------|---------|
| `-DWITH_CPU=ON` | CPU (OpenMP) |
| `-DWITH_NVIDIA=ON` | NVIDIA CUDA (requires CUDAToolkit) |
| `-DWITH_ILUVATAR=ON` | Iluvatar (clang++ with `-x ivcore`) |
| `-DWITH_METAX=ON` | MetaX (requires `$MACA_PATH`) |
| `-DWITH_CAMBRICON=ON` | Cambricon (requires `$NEUWARE_HOME`) |

`WITH_NVIDIA` and `WITH_ILUVATAR` cannot both be ON at the same time.

## Testing

```bash
pytest tests/                            # run all tests
pytest tests/test_add.py                 # run one test file
pytest tests/test_add.py::test_add       # run a single test
pytest tests/ --benchmark                # run with performance benchmarks
pytest tests/ -v --tb=short             # verbose output
```

Tests auto-parametrize on `dtype` (float32/float16/bfloat16) and `device` (cpu, and cuda/mlu if available). Tests import `infini.ops`, so the package must be installed (or built and on `PYTHONPATH`).

## Linting

```bash
ruff check .
ruff format .
```

## Code Style

Follow PEP 8 as the primary style guide. For areas PEP 8 does not cover in detail, refer to the GDScript style guide for non-syntax conventions. Always run `ruff format && ruff check` before committing.

### Comments

- Comments must be complete English sentences: capitalize the first word, end with punctuation.
- Use Markdown backtick syntax for code references within comments (e.g. `` `variable_name` ``).
- Error messages and framework-conventional strings (e.g. `pytest.skip` reasons) follow their own conventions — typically lowercase, no trailing period.

### Docstrings

- Follow PEP 257. One-line docstrings stay on a single line. Multi-line docstrings have a summary line, a blank line, then the description.

### Blank lines

- No blank line between a function signature and its body when there is no docstring or comment.
- Add a blank line before and after `if`, `for`, `while`, and similar compound statements.
- Add a blank line before a `return` statement unless it is directly inside an `if`/`for`/`while` block body.

## CI

The `.ci/` directory implements a multi-platform, resource-aware CI system with Docker-based execution, GitHub integration, and cross-machine job dispatch.

### Configuration

`config.yaml` uses a **platform-centric** structure that normalizes to flat `{platform}_{job}` names at load time (e.g. `nvidia_gpu`). Each platform defines its Docker image, setup commands, volumes, env vars, and jobs. Jobs inherit platform-level defaults.

Supported platforms: **nvidia**, **iluvatar**, **ascend** (ascend not ready yet).

### Building images

```bash
python .ci/build.py --platform nvidia       # build one platform
python .ci/build.py --platform all          # build all platforms
python .ci/build.py --platform nvidia --force  # skip Dockerfile change detection
python .ci/build.py --push --dry-run        # push to registry (preview)
```

Dockerfiles live in `.ci/images/{platform}/Dockerfile`. Proxy variables from the host are forwarded automatically.

### Running the pipeline locally

```bash
python .ci/run.py                                   # auto-detect platform, run all jobs
python .ci/run.py --job gpu --stage test             # run specific job/stage
python .ci/run.py --job gpu --gpu-id 0,2             # override GPU allocation
python .ci/run.py --image-tag stable                 # use a specific image tag
python .ci/run.py --dry-run                          # preview docker commands
```

Platform is auto-detected by checking for `nvidia-smi` or `ixsmi` on PATH.

### Agent (scheduler + webhook server)

`agent.py` provides a resource-aware scheduler with GitHub webhook support and REST API:

```bash
# Start the agent (webhook server + scheduler)
python .ci/agent.py serve --port 8080 --webhook-secret <secret>

# Dispatch jobs to remote agents via HTTP
python .ci/agent.py run --branch feat/xxx --platform nvidia
python .ci/agent.py run --job nvidia_gpu --dry-run
```

**Key capabilities:**

- **Resource-aware scheduling** — dynamically allocates GPUs based on utilization threshold; queues jobs when resources are busy.
- **GitHub webhooks** — triggers jobs on push/PR events (`/webhook` endpoint, HMAC-SHA256 verified).
- **REST API** — `/api/run` (trigger jobs, Bearer token auth), `/api/job/{id}` (query status), `/status` (queue + resources), `/health`.
- **GitHub commit status** — reports pending/success/failure per job via `github_status.py`.
- **Cross-machine dispatch** — sends jobs to remote platform agents and polls for results.

### Module overview

| File | Purpose |
|------|---------|
| `config.yaml` | Platform-centric CI configuration |
| `build.py` | Docker image builder with change detection |
| `run.py` | Standalone Docker CI runner (clone, setup, stages) |
| `agent.py` | Scheduler, webhook server, remote dispatch CLI |
| `utils.py` | Config normalization (`normalize_config`), git helpers |
| `ci_resource.py` | GPU/memory detection and thread-safe allocation (`ResourcePool`) |
| `github_status.py` | GitHub Commit Status API wrapper (zero external deps) |

### Tests

```bash
pytest .ci/tests/                          # run all CI tests
pytest .ci/tests/test_agent.py             # test scheduler and webhooks
```

## Architecture

### C++ layer (`src/`)

- **`src/base/<op>.h`** — Abstract base class for each operator (e.g. `Add`, `Gemm`, `RmsNorm`). Declares the constructor (capturing tensor metadata) and a pure-virtual `operator()`.
- **`src/<backend>/<op>.*`** — Backend-specific specializations: `src/cpu/`, `src/cuda/`, `src/nvidia/`, `src/metax/`, `src/cambricon/`, `src/iluvatar/`. Each provides `template<> class Operator<Add, Device::Type::kNvidia>`.
- **`src/operator.h`** — `Operator<Key, Device>` template that dispatches to the correct device specialization at `make()` time via `DispatchFunc`. Also caches constructed operator descriptors keyed on tensor shape/dtype/strides.
- **`src/tensor.h` / `src/device.h` / `src/data_type.h`** — Core data model: `Tensor` (pointer + shape + strides + dtype + device), `Device`, `DataType`.
- **`src/dispatcher.h`** — `DispatchFunc` selects the right device at runtime based on `Device::Type` and the compile-time `ActiveDevices` set.

### Python bindings

Python bindings are **auto-generated** by `scripts/generate_wrappers.py` using libclang to parse `src/base/<op>.h`. The generated output lands in `generated/bindings/ops.cc` and `generated/include/`. Bindings expose each operator both as a callable class (stateful, with constructor) and as a free function (`infini.ops.add(input, other, out)`).

### Test framework (`tests/`)

- `conftest.py` implements the `@pytest.mark.auto_act_and_assert` marker: the test function returns a `Payload(func, ref, args, kwargs, rtol, atol)` and the framework calls both, clones tensors for the reference, and asserts `torch.allclose`.
- `device` and `dtype` fixtures are auto-parametrized in `conftest.py`; individual tests can override with explicit `@pytest.mark.parametrize`.
- `tests/utils.py` provides `randn_strided`, `randint_strided`, `empty_strided`, `clone_strided` to create tensors with arbitrary strides.

### Adding a new operator

1. Create `src/base/<op>.h` with an abstract class inheriting `Operator<OpName>`.
2. Implement backend specializations in `src/<backend>/`.
3. Re-run `scripts/generate_wrappers.py` (or rebuild with `GENERATE_PYTHON_BINDINGS=ON`) to regenerate Python bindings.
4. Add a `tests/test_<op>.py` using the `Payload` / `auto_act_and_assert` pattern.
