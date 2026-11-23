# Agent Guide

## Start Here

- Read [CONTRIBUTING.md](CONTRIBUTING.md) for coding style, operator conventions,
  commits, and pull request requirements. Keep changes focused on the task.
- Use [docs/README.md](docs/README.md) as the documentation index and
  [docs/build.md](docs/build.md) for build configuration. This file supplements
  those guides; keep detailed policies in their existing documents.
- Before adding or changing a public operator interface, read
  [docs/operator-api-alignment.md](docs/operator-api-alignment.md). Name the
  upstream contract and document any C++ adaptations; matching the mathematical
  operation alone does not establish API compatibility.

## Repository Map

- `src/base/`: operator interfaces and shared semantics.
- `src/native/`: native implementations. `src/native/cuda/` contains shared
  CUDA-like implementations and platform-specific subdirectories; other native
  backends have their own directories under `src/native/`.
- `src/torch/ops/`, `src/linked/`, `src/triton/`, and `src/ninetoothed/`:
  alternative implementation backends. See [docs/backends.md](docs/backends.md).
- `tests/`: pytest coverage and reference comparisons. Follow nearby tests and
  the test conventions in `CONTRIBUTING.md`.
- `scripts/generate_wrappers.py`, `scripts/generate_torch_ops.py`,
  `scripts/torch_ops.yaml`, and `scripts/resolve_linked_ops.py`: generation and
  provider-selection inputs. Change the relevant source or generator, rather
  than hand-editing output under `generated/`; regenerate through the build.
- `.github/workflows/ci_test.yml` and `.github/ci_config.yml`: CI entry points.
  The workflow pins an external CI implementation; consult that revision's
  documentation before using its command-line tools.

## Implementation Constraints

- Runtime, device, and core dtype definitions come from InfiniRT. The local
  `src/runtime.h`, `src/device.h`, and `src/data_type.h` expose aliases and
  integration helpers. Inspect that boundary before changing runtime behavior.
- Resolve native element types with the appropriate device, for example
  `TypeMapType<device, dtype>`. Half and bfloat16 representations and operations
  can differ between platforms.
- Keep vendor headers and intrinsics in the appropriate platform layer.
  Shared CUDA-like templates must remain valid for the backends that include
  them; use existing device-specific caster and specialization patterns.
- Include the headers a file needs directly. Separate kernels from launchers
  according to `CONTRIBUTING.md` and the surrounding implementation.
- Preserve the operator's shape, stride, dtype, device, output, aliasing, and
  mutation contract. For behavior changes, add or update focused reference
  comparisons, including relevant noncontiguous layouts and boundary cases.

## Build and Validate

Install InfiniRT first and configure its install prefix via `INFINI_RT_ROOT`,
as described in [README.md](README.md) and `docs/build.md`. Set up the development
dependencies and the selected backend's SDK before building.

The following commands run from the repository root in a configured Bash
environment. Replace `nvidia` with the affected platform supported by
`scripts/dev/test.sh`:

```bash
bash scripts/dev/test.sh nvidia --smoke
```

This builds and installs the smoke subset, then runs smoke tests. It does not
cover every operator. For a changed operator, also run its relevant tests with
the operator included in the build, for example:

```bash
bash scripts/dev/test.sh nvidia -- tests/test_add.py
```

The second command builds without smoke mode before testing. `--no-build` is
available only when deliberately reusing an up-to-date install containing the
required operators. Check `docs/build.md` for operator allowlists that can
further restrict a build.

Follow `CONTRIBUTING.md` for validation scope: smoke build and tests on each
affected platform, with full-suite or full-platform validation for high-risk
changes and changes to shared build, dispatch, generation, or cross-platform
behavior. For documentation-only edits, check links, commands, and whitespace.

## Review Handoff

- Summarize the behavior changed, affected backends, and compatibility impact.
- Record commands actually run, platforms, results, failures, and skipped or
  unavailable validation. Do not infer platform coverage from another device's
  successful run. Include reproducible before/after evidence for performance
  claims.
- Use [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) when
  preparing a PR. Keep unrelated cleanup and generated build artifacts out of
  the change.
