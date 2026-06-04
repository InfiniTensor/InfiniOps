# Infini Ops Plugin Contract v1

This contract defines the build-time plugin boundary for `infini::ops`. It is a source-level integration contract, not a runtime loading ABI. v1 still builds one `libinfiniops.so` and the existing Python `ops` module. Plugins only decide which platform sources, operator roots, device headers, CMake entries, and test device mappings participate in that build.

## Scope

- v1 plugins are loaded during configure and code generation. They do not use `dlopen`, do not publish a stable binary ABI, and do not change the public C++ or Python operator API.
- v1 only accepts device names already known by core: `cpu`, `nvidia`, `iluvatar`, `hygon`, `metax`, `moore`, `cambricon`, and `ascend`. External plugins cannot add a new `Device::Type` yet.
- `kind=shared` is for shared source layers such as `cuda-common`. Shared plugins do not declare user-visible devices.
- `kind=device` is for actual device platforms. Device plugins must declare at least one device and must provide header and test mappings for each declared device.

## Manifest

Each built-in plugin owns a `plugin.json` under `plugins/<name>/`. The manifest must include these fields:

- `name`: plugin name. It must match the `plugins/<name>/` directory name.
- `kind`: either `shared` or `device`.
- `contract_version`: currently `1`.
- `devices`: device names implemented by this plugin. Use `[]` for `shared` plugins.
- `depends`: plugin names that must be enabled before this plugin.
- `cmake_entry`: CMake entry file, relative to the plugin directory.
- `source_roots`: source roots scanned by the build for enabled plugins.
- `operator_roots`: operator implementation roots scanned by wrapper generation.
- `device_headers`: mapping from owned device name to the corresponding device header.
- `test_devices`: mapping from owned device name to the existing `pytest --devices` selector.

Example:

```json
{
  "name": "cpu",
  "kind": "device",
  "contract_version": 1,
  "devices": ["cpu"],
  "depends": [],
  "cmake_entry": "plugin.cmake",
  "source_roots": ["src/native/cpu"],
  "operator_roots": ["src/native/cpu/ops"],
  "device_headers": {"cpu": "native/cpu/device_.h"},
  "test_devices": {"cpu": "cpu"}
}
```

## Path Rules

Manifest paths are logical repository paths unless noted otherwise. `source_roots`, `operator_roots`, and `device_headers` values must be relative paths, must be non-empty strings, and must not contain `..` components. `cmake_entry` is relative to the plugin directory and must stay inside that plugin directory. Absolute paths are rejected.

Use Markdown code spans in diagnostics when naming fields, paths, CMake options, or generated files, for example `cmake_entry`, `plugins/cpu/plugin.json`, and `INFINI_OPS_PLUGINS`.

## CMake Contract

Core exposes internal CMake APIs for built-in plugins:

- `infini_ops_register_plugin`: registers plugin metadata and source roots.
- `infini_ops_register_device`: registers one device owned by a device plugin.
- `infini_ops_enable_plugin`: enables one plugin and its dependencies.
- `infini_ops_enable_requested_plugins`: resolves the requested plugin set during configure.
- `infini_ops_write_plugin_registry`: writes the build registry consumed by code generation.

Platform entries should keep platform-specific compile definitions, source lists, and link libraries in their own `plugin.cmake` files. Core CMake should consume registered data instead of hard-coding per-platform globbing or link rules.

## Selection

`INFINI_OPS_PLUGINS` is the canonical configure option for selecting plugins. Legacy options such as `WITH_NVIDIA`, `WITH_ASCEND`, and `WITH_CAMBRICON` remain compatibility shims and map onto the same plugin enable path.

`cuda-common` is a `shared` plugin. It is enabled through dependencies from CUDA-like device plugins such as `nvidia`, `iluvatar`, `metax`, `moore`, and `hygon`; it does not expose a standalone user device.

## Code Generation

The build writes a plugin registry after dependency resolution. `scripts/generate_wrappers.py` reads that registry and scans only enabled `operator_roots` and `device_headers`. The existing `Operator<Op, Device::Type, implementation_index>` implementation model is preserved in v1.

## Testing

`test_devices` preserves the current `pytest --devices <platform>` workflow. A device plugin change should normally run tests for its mapped devices. Changes to core, this contract, code generation, or shared plugins such as `cuda-common` can affect multiple devices and should trigger broader compatibility testing.
