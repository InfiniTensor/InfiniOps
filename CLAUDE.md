# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InfiniOps is a high-performance, hardware-agnostic operator library for LLM inference. It provides optimized operator implementations across multiple hardware accelerators (CPU, NVIDIA, Cambricon, Ascend, Iluvatar, MetaX, Moore, Hygon) using C++17 templates with compile-time dispatch.

## Build Commands

```bash
# CPU only (default when no accelerator specified)
mkdir build && cd build
cmake .. -DWITH_CPU=ON
make -j$(nproc)

# NVIDIA GPU
cmake .. -DWITH_NVIDIA=ON -DWITH_CPU=ON
make -j$(nproc)

# With Python bindings (pip install)
pip install . -C cmake.define.WITH_CPU=ON
pip install . -C cmake.define.WITH_NVIDIA=ON

# Auto-detect devices + Python bindings
pip install .[dev]
```

Backend options are mutually exclusive for GPU types: only one of `WITH_NVIDIA`, `WITH_ILUVATAR`, `WITH_METAX`, `WITH_MOORE`, `WITH_HYGON` can be ON at a time. CPU can be combined with any GPU backend.

## Testing

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_add.py

# Run with benchmark mode
pytest tests/ --benchmark

# Parallel execution
pytest tests/ -n 8
```

Tests use PyTorch tensors as reference implementations. Test files are auto-parametrized by `dtype` (float32/float16/bfloat16) and `device` (auto-detected from available hardware). The `@pytest.mark.auto_act_and_assert` pattern runs both the operator and PyTorch reference, then compares with `torch.allclose`.

## Architecture

### Operator Lifecycle (the CRTP pattern)

Every operator follows the same inheritance chain:

```
OperatorBase                      → handle/stream/workspace management
  └── Operator<Key>               → make()/call()/caching (CRTP)
        └── BaseClass (e.g., Add) → interface + metadata
              └── CudaImpl<Backend> → platform-generic GPU impl
                    └── Operator<Key, DeviceType> → platform specialization
```

- **`src/operator.h`**: Core `Operator<Key>` template. `call()` caches operators by CacheKey (shape/dtype/stride). `make()` dispatches to device-specific specialization via `DispatchFunc<ActiveDevices>`.
- **`src/base/*.h`**: Base classes defining operator interface. Constructor caches tensor metadata (shape, strides, dtype, contiguity). Pure virtual `operator()` for execution.
- **`src/cuda/*/kernel.h`**: CUDA implementations parameterized by `Backend` (abstracting malloc/memcpy/free/stream_t).
- **`src/nvidia/*/kernel.h`**: NVIDIA specialization defining `NvidiaBackend` and `Operator<Op, kNvidia>`.

### Adding a new operator

1. Create `src/base/<op>.h` — base class inheriting `Operator<OpName>`, constructor caches metadata, pure virtual `operator()`.
2. Create `src/cuda/<op>/kernel.h` — template `Cuda<Op><Backend>` inheriting base, implements `operator()`.
3. Create `src/nvidia/<op>/kernel.h` — define `NvidiaBackend`, specialize `Operator<OpName, kNvidia>`.
4. Other CUDA-compatible platforms (Iluvatar, MetaX, Moore, Hygon) reuse `src/cuda/` code with their own Backend.
5. Add tests in `tests/test_<op>.py`.

### Adding a new platform

1. Add `option(WITH_<PLATFORM>)` to top-level `CMakeLists.txt`.
2. Add device type to `src/device.h` (Device::Type enum + EnabledDeviceFilter).
3. Add source patterns to `src/CMakeLists.txt`.
4. Create `src/<platform>/` with Backend struct and `Operator<Key, DeviceType>` specializations.

### Dispatch mechanism (`src/dispatcher.h`)

`DispatchFunc<TypeList>(runtime_value, lambda)` generates a compile-time switch-case from a type list. Used to dispatch by device type (at `make()` time) and by data type (at kernel launch time). The `EnabledDeviceFilter` in `device.h` uses `#ifdef` macros to control which device types are compiled in.

### Key types

- **`Tensor`** (`src/tensor.h`): Carries data pointer, shape, strides, dtype, device. Not owning.
- **`DataType`** (`src/data_type.h`): Enum with `TypeMap<Dtype>→C++ type` and `DataTypeMap<C++ type>→Dtype` mappings.
- **`Device`** (`src/device.h`): Enum of supported hardware types (kCpu, kNvidia, kCambricon, kAscend, kMetax, kMoore, kIluvatar, kHygon, etc.).

## Code Conventions

- **Namespace**: `infini::ops`
- **Member variables**: trailing underscore (`stream_`, `dtype_`)
- **Enum values**: `kCamelCase` (`kFloat32`, `kNvidia`)
- **Optional parameters**: Use `std::optional`, with `value_or()` defaults in constructor (see `base/gemm.h` as reference).
- **Protected members**: Base classes expose cached metadata as `protected` for subclass access.
- **Backend pattern**: Platform-specific APIs abstracted into Backend structs with `malloc`, `memcpy`, `free`, `stream_t` (see `nvidia/add/kernel.h`).

## Supported Devices

| Backend | Compile Flag | Notes |
|---------|-------------|-------|
| CPU | `WITH_CPU` | OpenMP, default if no GPU |
| NVIDIA | `WITH_NVIDIA` | CUDA |
| Cambricon | `WITH_CAMBRICON` | CNNL/BANG |
| Ascend | `WITH_ASCEND` | ACLNN |
| Iluvatar | `WITH_ILUVATAR` | CUDA-compatible via clang++ |
| MetaX | `WITH_METAX` | MACA, uses mxcc_wrapper.sh |
| Moore | `WITH_MOORE` | MUSA, uses mcc_wrapper.sh |
| Hygon | `WITH_HYGON` | CUDA-compatible |
