# AGENTS.md

## Project Overview

InfiniOps is a cross-platform operator library supporting multiple backends: **CPU**, **Nvidia**, **MetaX**, **Iluvatar**, **Moore**, **Cambricon**, and more. The codebase uses a template-based architecture where each platform provides its own specialization.

## Build & Test

```bash
python -m pip install .[dev]
python -m pytest -n 1
```

To run CI across all supported platforms on remote machines:

```bash
python .ci/agent.py run --branch <branch-name>
```

To run local code on the current machine:

```bash
python .ci/run.py --local
```

CI must pass on **all** platforms. Build failures are blockers; test failures may be pre-existing.

## Code Architecture

### `Runtime<Device::Type>` Template

`Runtime<Device::Type>` is the unified runtime abstraction for all platforms. Each specialization lives in `<platform>/runtime_.h` and wraps the platform's native APIs with a consistent interface:

- `Stream` — type alias for the platform's stream/queue type (e.g. `cudaStream_t`, `cnrtQueue_t`).
- `kDeviceType` — `static constexpr Device::Type` identifying the platform.
- `Malloc`, `Free`, `Memcpy`, `MemcpyHostToDevice` — thin wrappers around the platform's runtime APIs.

**`Runtime`, `Blas`, and similar classes/structs must only contain direct bindings to platform APIs** (e.g. what you find in `cuda_runtime.h` or `cublas_v2.h`). Helpers like `GetOptimalBlockSize`, `GetDataType`, or `GetComputeType` must live in separate classes — they are derived logic, not API bindings.

### CRTP Interface Enforcement

`Runtime` specializations inherit from a CRTP base to declare their interface level, then call `static_assert(Runtime<...>::Validate())` after the struct is complete. This catches signature mismatches at compile time, analogous to the `override` keyword.

- `RuntimeBase<Derived>` — requires `kDeviceType` only (e.g. CPU).
- `DeviceRuntime<Derived>` — additionally requires `Stream` and more.

### Device-Aware Type System

The type system is split across several layers:

- **`data_type.h`** — declares `TypeMap<dev, dtype>`, `IsFP16<kDev, T>`, `IsBFloat16<kDev, T>`, and common type lists.
- **`<platform>/data_type_.h`** — provides `TypeMap` specializations mapping `kFloat16`/`kBFloat16` to platform-specific C++ types (e.g. `half`, `__nv_bfloat16`). Also defines platform-specific type aliases like `cuda_bfloat16`.
- **`caster.h`** — forward-declares `Caster<Device::Type>`.
- **`<platform>/caster.cuh`** — specializes `Caster` for each platform. CUDA-like platforms inherit from `CudaCasterImpl` in `cuda/caster.cuh`.
- **`dispatcher.h`** — `DispatchFunc` for device-aware runtime dispatch over type and block-size lists.

Use `IsFP16<kDev, T>` / `IsBFloat16<kDev, T>` instead of `std::is_same_v<T, half>` in shared headers. Use `Caster<kDev>::template Cast<T>()` for type conversions.

### Include-Order Independence

Shared headers under `src/cuda/` must **not** reference platform-specific types or intrinsics directly (e.g. `half`, `__float2half`, `cuda_bfloat16`). Instead, use dependent template expressions like `Caster<kDev>::template Cast<T>()` so that name lookup is deferred to instantiation time (C++ two-phase lookup). This ensures shared headers are self-contained and include-order-independent.

### Platform-Specific Directory Structure

Platform-specific code **must** stay in its own directory under `src/<platform>/`. Do **not** lift platform-specific files (e.g. `device_.h`, casters, type mappings) into a common directory. The key architectural principle of the device-aware refactoring is separation into platform-specific directories.

### Moore Polyfills

Moore (MUSA) lacks some CUDA intrinsics. `src/moore/polyfills.cuh` provides replacements (e.g. `hrcp`, `__hadd`). This file must include `<musa_fp16.h>` and `<musa_bf16.h>` **before** defining polyfill macros, to prevent macro collisions with system headers.

## Code Style

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full style guide. Key rules for agents:

- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) strictly. Use the default `.clang-format`.
- Keep changes minimal — do not add what is not necessary.
- Do not use exceptions. Use `assert` for error handling.
- Use absolute includes (e.g. `#include "cambricon/common.h"`) instead of relative includes.
- Include guards should match the filename exactly, including underscores (e.g. `device_.h` -> `INFINI_OPS_<PLATFORM>_DEVICE__H_` with double underscore).
- Wrap platform includes that need ordering protection with `// clang-format off` / `// clang-format on`.
- Leave a blank line between consecutive `using` type alias declarations.
- One blank line between classes, functions, and members within a class.
- Comments and error messages must be in English, use complete sentences (capitalized, ending with punctuation), and use Markdown backticks around identifiers.
- Error messages follow [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html#error-and-warning-messages).
- Kernel file naming: `kernel` for custom kernels, `kernel_v2`/`kernel_v3` for variants, algorithm name for well-known algorithms (e.g. `flash_attention_v2`), library name for library-based implementations (e.g. `blas`).
- Separate kernel (`.cuh`) from kernel launcher (`.h`).

## Commit & PR Conventions

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).
- Concise single-line commit messages with Markdown backticks around identifiers.
- Each commit should do a single job — keep commits logically separated.
- PR titles follow the same Conventional Commits format.
- Small PRs should be squashed; large PRs may keep multiple commits if each is meaningful.
- Branch names: `<type>/xxx-yyyy-zzzz` (hyphen-separated, type matches PR title).
- Do **not** include AI as a co-author.

## Known Pitfalls

- **Non-dependent name lookup in NVCC**: Direct CUDA intrinsics like `__float2half()` are non-dependent names resolved at template definition time, before platform headers may be available. Wrap them in `Caster<kDev>` to make them dependent.
- **Moore `hrcp` macro**: The `#define hrcp infini::ops::hrcp` in `polyfills.cuh` will collide with `musa_fp16_mtgpu.h` if the MUSA FP16 header is included after the macro definition.
