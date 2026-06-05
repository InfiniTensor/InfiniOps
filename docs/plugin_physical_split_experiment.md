# Infini Ops Physical Split Experiment

This note records a local experiment for splitting a device/operator plugin outside the main `InfiniOps` repository while still building one `libinfiniops.so`.

## Layout

The external plugin root was created under `/tmp/infini-ops-physical-split/plugins`. The plugin directory was `/tmp/infini-ops-physical-split/plugins/external-cpu-add` and contained:

- `plugin.json`
- `plugin.cmake`
- `ops/add/add.h`

`ops/add/add.h` was copied from `src/native/cpu/ops/add/add.h` to simulate a physically split operator implementation.

The manifest kept source paths relative to the external plugin directory:

```json
{
  "name": "external-cpu-add",
  "kind": "device",
  "contract_version": 1,
  "devices": ["cpu"],
  "depends": [],
  "cmake_entry": "plugin.cmake",
  "source_roots": ["ops"],
  "operator_roots": ["ops"],
  "device_headers": {"cpu": "native/cpu/device_.h"},
  "test_devices": {"cpu": "cpu"}
}
```

The CMake entry registered resolved external paths for build/codegen consumption:

```cmake
infini_ops_register_device(
    NAME external-cpu-add
    CMAKE_ENTRY plugin.cmake
    DEVICES cpu
    SOURCE_ROOTS "/tmp/infini-ops-physical-split/plugins/external-cpu-add/ops"
    OPERATOR_ROOTS "/tmp/infini-ops-physical-split/plugins/external-cpu-add/ops"
    DEVICE_HEADERS cpu=native/cpu/device_.h
    TEST_DEVICES cpu=cpu)

target_compile_definitions(infiniops PUBLIC WITH_CPU=1)
find_package(OpenMP REQUIRED COMPONENTS CXX)
target_link_libraries(infiniops PRIVATE OpenMP::OpenMP_CXX)
```

## Commands

Configure with the external plugin root and generated operator call instantiations enabled:

```sh
cmake -S . -B build/physical-split-external-add \
  -DINFINI_OPS_PLUGINS=external-cpu-add \
  -DINFINI_OPS_PLUGIN_ROOTS=/tmp/infini-ops-physical-split/plugins \
  -DGENERATE_PYTHON_BINDINGS=OFF \
  -DGENERATE_OPERATOR_CALL_INSTANTIATIONS=ON
```

Build the main library:

```sh
cmake --build build/physical-split-external-add --target infiniops --parallel 2
```

## Result

The configure step loaded the external plugin and generated wrappers successfully:

```text
-- infini_ops plugins: `external-cpu-add`
-- infini_ops plugin devices: `cpu`
-- Generating wrappers - done
```

The generated registry pointed at the external operator root:

```json
{
  "plugins": ["external-cpu-add"],
  "devices": ["cpu"],
  "source_roots": ["/tmp/infini-ops-physical-split/plugins/external-cpu-add/ops"],
  "operator_roots": ["/tmp/infini-ops-physical-split/plugins/external-cpu-add/ops"],
  "device_headers": {"cpu": "native/cpu/device_.h"},
  "test_devices": {"cpu": "cpu"}
}
```

Generated sources included the external implementation header directly:

```cpp
#include "/tmp/infini-ops-physical-split/plugins/external-cpu-add/ops/add/add.h"
```

The build completed and produced `build/physical-split-external-add/src/libinfiniops.so`. Symbol inspection found `Operator<infini::ops::Add, Device::Type::kCpu, 0>` and generated `Add` call instantiation symbols in that library.

## Findings

- Physical source split is viable for header-only operator implementations with the current build-time plugin mechanism.
- The external plugin can keep `plugin.json` paths relative for contract validation, while `plugin.cmake` registers resolved paths for CMake/codegen.
- The external plugin is responsible for platform-specific compile definitions and link libraries, such as `WITH_CPU=1` and `OpenMP::OpenMP_CXX` in this experiment.
- The current generated code uses absolute includes for external operator headers. This works for local builds, but it is not ideal for installed build trees or reproducible generated artifacts. A future hardening step should add a stable external include-root mapping instead of embedding `/tmp/...` paths.
- v1 still cannot add a new `Device::Type`; the external plugin reused the existing `cpu` device.
