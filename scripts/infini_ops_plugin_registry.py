import json
import pathlib

KNOWN_DEVICES = {
    "cpu",
    "nvidia",
    "iluvatar",
    "hygon",
    "metax",
    "moore",
    "cambricon",
    "ascend",
}

REQUIRED_FIELDS = {
    "name",
    "kind",
    "contract_version",
    "devices",
    "depends",
    "cmake_entry",
    "source_roots",
    "operator_roots",
    "device_headers",
    "test_devices",
}


def _as_list(value, field, plugin_name):
    if not isinstance(value, list):
        raise ValueError(f"plugin `{plugin_name}` field `{field}` must be a `list`")

    return value


def _as_dict(value, field, plugin_name):
    if not isinstance(value, dict):
        raise ValueError(f"plugin `{plugin_name}` field `{field}` must be a `dict`")

    return value


def _load_manifest(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_FIELDS.difference(data)

    if missing:
        raise ValueError(
            f"plugin manifest `{path}` is missing required fields: "
            f"{', '.join(sorted(missing))}"
        )

    name = data["name"]

    if name != path.parent.name:
        raise ValueError(
            f"plugin manifest `{path}` declares name `{name}`, "
            f"expected `{path.parent.name}`"
        )

    if data["kind"] not in {"shared", "device"}:
        raise ValueError(f"plugin `{name}` has invalid kind `{data['kind']}`")

    if data["contract_version"] != 1:
        raise ValueError(
            f"plugin `{name}` uses unsupported contract version "
            f"`{data['contract_version']}`"
        )

    cmake_entry = data["cmake_entry"]
    if not isinstance(cmake_entry, str) or not cmake_entry:
        raise ValueError(f"plugin `{name}` field `cmake_entry` must be a `string`")

    if not (path.parent / cmake_entry).is_file():
        raise ValueError(
            f"plugin `{name}` `CMake` entry `{cmake_entry}` was not found"
        )

    devices = _as_list(data["devices"], "devices", name)
    depends = _as_list(data["depends"], "depends", name)
    _as_list(data["source_roots"], "source_roots", name)
    _as_list(data["operator_roots"], "operator_roots", name)
    device_headers = _as_dict(data["device_headers"], "device_headers", name)
    test_devices = _as_dict(data["test_devices"], "test_devices", name)

    for device in devices:
        if device not in KNOWN_DEVICES:
            raise ValueError(f"plugin `{name}` declares unknown device `{device}`")

    for device in device_headers:
        if device not in devices:
            raise ValueError(
                f"plugin `{name}` has device header for non-owned device `{device}`"
            )

    for device in test_devices:
        if device not in devices:
            raise ValueError(
                f"plugin `{name}` has test device for non-owned device `{device}`"
            )

    if data["kind"] == "device" and not devices:
        raise ValueError(f"device plugin `{name}` must declare at least one device")

    if data["kind"] == "shared" and devices:
        raise ValueError(f"shared plugin `{name}` must not declare devices")

    for dependency in depends:
        if not isinstance(dependency, str):
            raise ValueError(f"plugin `{name}` dependency names must be `string`s")

    return data


def _append_unique(values, new_values):
    for value in new_values:
        if value not in values:
            values.append(value)


def load_plugin_registry(plugin_root, requested_plugins):
    plugin_root = pathlib.Path(plugin_root)
    manifests = {
        path.parent.name: _load_manifest(path)
        for path in sorted(plugin_root.glob("*/plugin.json"))
    }

    ordered = []
    visiting = []
    visited = set()

    def visit(name):
        if name in visited:
            return

        if name in visiting:
            cycle = " -> ".join([*visiting, name])
            raise ValueError(f"plugin dependency cycle detected: {cycle}")

        if name not in manifests:
            raise ValueError(f"requested plugin `{name}` was not found")

        visiting.append(name)
        for dependency in manifests[name]["depends"]:
            visit(dependency)
        visiting.pop()
        visited.add(name)
        ordered.append(name)

    for name in requested_plugins:
        visit(name)

    devices = []
    source_roots = []
    operator_roots = []
    device_headers = {}
    test_devices = {}

    for name in ordered:
        manifest = manifests[name]
        _append_unique(devices, manifest["devices"])
        _append_unique(source_roots, manifest["source_roots"])
        _append_unique(operator_roots, manifest["operator_roots"])
        device_headers.update(manifest["device_headers"])
        test_devices.update(manifest["test_devices"])

    return {
        "plugins": ordered,
        "devices": devices,
        "source_roots": source_roots,
        "operator_roots": operator_roots,
        "device_headers": device_headers,
        "test_devices": test_devices,
    }


def write_plugin_registry(path, registry):
    pathlib.Path(path).write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
