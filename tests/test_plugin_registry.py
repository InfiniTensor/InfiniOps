import importlib.util
import json
import pathlib
import sys


def _load_registry_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "infini_ops_plugin_registry.py"
    )
    spec = importlib.util.spec_from_file_location("infini_ops_plugin_registry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def _write_manifest(root, name, data, create_cmake_entry=True):
    plugin_dir = root / name
    plugin_dir.mkdir()
    path = plugin_dir / "plugin.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    if create_cmake_entry and data.get("cmake_entry"):
        (plugin_dir / data["cmake_entry"]).write_text(
            "# test plugin entry\n", encoding="utf-8"
        )

    return path


def test_load_plugins_orders_dependencies_and_merges_device_metadata(tmp_path):
    module = _load_registry_module()
    plugin_root = tmp_path / "plugins"
    plugin_root.mkdir()
    _write_manifest(
        plugin_root,
        "cuda-common",
        {
            "name": "cuda-common",
            "kind": "shared",
            "contract_version": 1,
            "devices": [],
            "depends": [],
            "cmake_entry": "plugin.cmake",
            "source_roots": ["src/native/cuda"],
            "operator_roots": ["src/native/cuda/ops"],
            "device_headers": {},
            "test_devices": {},
        },
    )
    _write_manifest(
        plugin_root,
        "nvidia",
        {
            "name": "nvidia",
            "kind": "device",
            "contract_version": 1,
            "devices": ["nvidia"],
            "depends": ["cuda-common"],
            "cmake_entry": "plugin.cmake",
            "source_roots": ["src/native/cuda/nvidia"],
            "operator_roots": ["src/native/cuda/nvidia/ops"],
            "device_headers": {"nvidia": "native/cuda/nvidia/device_.h"},
            "test_devices": {"nvidia": "cuda"},
        },
    )

    registry = module.load_plugin_registry(plugin_root, ["nvidia"])

    assert registry["plugins"] == ["cuda-common", "nvidia"]
    assert registry["devices"] == ["nvidia"]
    assert registry["device_headers"] == {
        "nvidia": "native/cuda/nvidia/device_.h",
    }
    assert registry["operator_roots"] == [
        "src/native/cuda/ops",
        "src/native/cuda/nvidia/ops",
    ]
    assert registry["test_devices"] == {"nvidia": "cuda"}


def test_load_plugins_rejects_missing_cmake_entry(tmp_path):
    module = _load_registry_module()
    plugin_root = tmp_path / "plugins"
    plugin_root.mkdir()
    _write_manifest(
        plugin_root,
        "cpu",
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
            "test_devices": {"cpu": "cpu"},
        },
        create_cmake_entry=False,
    )

    try:
        module.load_plugin_registry(plugin_root, ["cpu"])
    except ValueError as exc:
        assert "`CMake` entry" in str(exc)
    else:
        raise AssertionError("manifest with missing `CMake` entry should be rejected")


def test_load_plugins_rejects_unknown_devices(tmp_path):
    module = _load_registry_module()
    plugin_root = tmp_path / "plugins"
    plugin_root.mkdir()
    _write_manifest(
        plugin_root,
        "unknown",
        {
            "name": "unknown",
            "kind": "device",
            "contract_version": 1,
            "devices": ["unknown"],
            "depends": [],
            "cmake_entry": "plugin.cmake",
            "source_roots": [],
            "operator_roots": [],
            "device_headers": {"unknown": "native/unknown/device_.h"},
            "test_devices": {"unknown": "unknown"},
        },
    )

    try:
        module.load_plugin_registry(plugin_root, ["unknown"])
    except ValueError as exc:
        assert "unknown device" in str(exc)
    else:
        raise AssertionError("unknown device manifest should be rejected")


def test_load_plugins_rejects_dependency_cycles(tmp_path):
    module = _load_registry_module()
    plugin_root = tmp_path / "plugins"
    plugin_root.mkdir()
    _write_manifest(
        plugin_root,
        "a",
        {
            "name": "a",
            "kind": "shared",
            "contract_version": 1,
            "devices": [],
            "depends": ["b"],
            "cmake_entry": "plugin.cmake",
            "source_roots": [],
            "operator_roots": [],
            "device_headers": {},
            "test_devices": {},
        },
    )
    _write_manifest(
        plugin_root,
        "b",
        {
            "name": "b",
            "kind": "shared",
            "contract_version": 1,
            "devices": [],
            "depends": ["a"],
            "cmake_entry": "plugin.cmake",
            "source_roots": [],
            "operator_roots": [],
            "device_headers": {},
            "test_devices": {},
        },
    )

    try:
        module.load_plugin_registry(plugin_root, ["a"])
    except ValueError as exc:
        assert "dependency cycle" in str(exc)
    else:
        raise AssertionError("cyclic plugin dependencies should be rejected")


def test_builtin_plugin_manifests_load_individually():
    module = _load_registry_module()
    plugin_root = pathlib.Path(__file__).resolve().parents[1] / "plugins"

    for plugin_name in (
        "cpu",
        "nvidia",
        "iluvatar",
        "hygon",
        "metax",
        "moore",
        "cambricon",
        "ascend",
    ):
        registry = module.load_plugin_registry(plugin_root, [plugin_name])

        assert plugin_name in registry["plugins"]
        assert registry["devices"] == [plugin_name]
        assert plugin_name in registry["device_headers"]
        assert plugin_name in registry["test_devices"]


def test_builtin_plugin_manifests_cover_cpu_and_cuda_common_dependencies():
    module = _load_registry_module()
    plugin_root = pathlib.Path(__file__).resolve().parents[1] / "plugins"

    cpu_registry = module.load_plugin_registry(plugin_root, ["cpu"])
    nvidia_registry = module.load_plugin_registry(plugin_root, ["nvidia"])

    assert cpu_registry["plugins"] == ["cpu"]
    assert cpu_registry["devices"] == ["cpu"]
    assert cpu_registry["device_headers"] == {"cpu": "native/cpu/device_.h"}
    assert nvidia_registry["plugins"][:2] == ["cuda-common", "nvidia"]
    assert nvidia_registry["devices"] == ["nvidia"]
    assert "src/native/cuda/ops" in nvidia_registry["operator_roots"]
    assert "src/native/cuda/nvidia/ops" in nvidia_registry["operator_roots"]
