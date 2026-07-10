# Build and Test

InfiniOps uses CMake, scikit-build-core, and Python packaging. The most common
entry is `python -m pip install` with CMake options passed through
`--config-settings`.

## Common CMake Options

| Option | Description | Default |
| --- | --- | :---: |
| `WITH_CPU` | Enable the CPU backend. | `OFF` |
| `WITH_NVIDIA` | Enable the NVIDIA CUDA backend. | `OFF` |
| `WITH_ILUVATAR` | Enable the Iluvatar CUDA-compatible backend. | `OFF` |
| `WITH_HYGON` | Enable the Hygon backend. | `OFF` |
| `WITH_METAX` | Enable the MetaX backend. | `OFF` |
| `WITH_CAMBRICON` | Enable the Cambricon backend. | `OFF` |
| `WITH_MOORE` | Enable the Moore backend. | `OFF` |
| `WITH_ASCEND` | Enable the Ascend backend. | `OFF` |
| `WITH_TORCH` | Enable PyTorch C++ ATen-backed operators. | `OFF` |
| `WITH_NINETOOTHED` | Enable NineToothed-generated kernels. | `OFF` |
| `AUTO_DETECT_DEVICES` | Auto-detect available device files. | `OFF` |
| `AUTO_DETECT_BACKENDS` | Auto-detect available backend packages. | `OFF` |
| `GENERATE_OPERATOR_CALL_INSTANTIATIONS` | Generate explicit C++ operator call instantiations. | `ON` |
| `GENERATE_PYTHON_BINDINGS` | Generate Python bindings. | `OFF` in raw CMake, `ON` in `pyproject.toml` |
| `INFINI_OPS_BUILD_DOCS` | Enable the Doxygen documentation target. | `OFF` |
| `INFINI_RT_ROOT` | InfiniRT install prefix containing `include/` and `lib/`. | `$INFINI_RT_ROOT` |
| `INFINI_OPS_SMOKE_BUILD` | Build only the smoke-test operator subset. | `OFF` |
| `INFINI_OPS_OPS` | Comma- or semicolon-separated operator allowlist. | empty |
| `INFINI_OPS_TORCH_OPS` | Comma- or semicolon-separated ATen operator allowlist. | empty |

Only one GPU backend should be enabled in a build. CPU may be enabled with the
selected accelerator backend.

## Python Wheel Build

Using CPU as the smallest backend:

```bash
python -m pip install .[dev] \
  --config-settings=cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix \
  --config-settings=cmake.define.WITH_CPU=ON
```

Using NVIDIA as an example accelerator backend:

```bash
python -m pip install .[dev] \
  --config-settings=cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix \
  --config-settings=cmake.define.WITH_CPU=ON \
  --config-settings=cmake.define.WITH_NVIDIA=ON
```

The built wheel installs the InfiniOps Python extension and the InfiniRT shared
library needed by the extension.

## Smoke Build

For routine development and pull requests, start with a smoke build:

```bash
python -m pip install .[dev] --no-build-isolation --no-deps \
  --config-settings=cmake.define.INFINI_RT_ROOT=/path/to/infini-rt-prefix \
  --config-settings=cmake.define.WITH_CPU=ON \
  --config-settings=cmake.define.INFINI_OPS_SMOKE_BUILD=ON
```

`INFINI_OPS_SMOKE_BUILD=ON` narrows generated wrappers, bindings, and generated
Torch ops to a representative operator subset. Use full builds for release
preparation, shared build or dispatch changes, wrapper generation changes, and
platform maintainer spot checks.

## Tests

Run the full test suite:

```bash
python -m pytest
```

Run the smoke set:

```bash
python -m pytest tests -m smoke -q
```

Select platforms explicitly:

```bash
python -m pytest tests -m smoke -q --devices cpu nvidia
```

The platform names accepted by the test harness include `nvidia`, `metax`,
`iluvatar`, `hygon`, `moore`, `cambricon`, and `ascend`. The harness maps those
names to the corresponding PyTorch device type when needed.

## Formatting

Run the checks that match the touched files:

```bash
ruff format --check .
ruff check .
```

C++ changes should also pass the repository `clang-format` and `clang-tidy`
expectations described in `CONTRIBUTING.md`.

## Documentation

Enable the Doxygen documentation target with:

```bash
cmake -S . -B build \
  -DINFINI_RT_ROOT=/path/to/infini-rt-prefix \
  -DWITH_CPU=ON \
  -DINFINI_OPS_BUILD_DOCS=ON
cmake --build build --target infiniops_docs
```

The generated HTML is written to `build/docs/reference/html`.

The Documentation Pages workflow uses the same target to validate pull requests
and publish `master` builds through GitHub Pages.

See [API Reference](api/reference.md) for reference scope and preview commands.
