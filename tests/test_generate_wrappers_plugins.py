import importlib.util
import pathlib
import sys
import types


def _load_wrapper_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_wrappers.py"
    )
    spec = importlib.util.spec_from_file_location("generate_wrappers_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    clang_module = types.ModuleType("clang")
    clang_cindex = types.ModuleType("clang.cindex")
    clang_cindex.CursorKind = types.SimpleNamespace(
        CONSTRUCTOR=object(),
        CXX_METHOD=object(),
    )
    clang_module.cindex = clang_cindex
    sys.modules.setdefault("clang", clang_module)
    sys.modules.setdefault("clang.cindex", clang_cindex)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def test_device_marker_headers_come_from_plugin_registry():
    module = _load_wrapper_module()
    registry = {
        "device_headers": {
            "cpu": "native/cpu/device_.h",
            "nvidia": "plugins/nvidia/include/device_.h",
        },
    }

    assert module._device_marker_headers(["nvidia"], registry) == [
        "plugins/nvidia/include/device_.h",
    ]


def test_get_all_ops_scans_plugin_operator_roots(tmp_path, monkeypatch):
    module = _load_wrapper_module()
    base_dir = tmp_path / "base"
    plugin_ops = tmp_path / "plugin" / "ops" / "demo"
    base_dir.mkdir()
    plugin_ops.mkdir(parents=True)
    (base_dir / "demo.h").write_text("class Demo {};\n", encoding="utf-8")
    impl_header = plugin_ops / "kernel.h"
    impl_header.write_text(
        "namespace infini::ops { class Operator<Demo, Device::Type::kCpu, 0> {}; }\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_BASE_DIR", base_dir)
    monkeypatch.setattr(module, "_GENERATED_BASE_DIR", tmp_path / "generated" / "base")

    ops = module._get_all_ops(
        ["cpu"],
        plugin_registry={"operator_roots": [str(tmp_path / "plugin" / "ops")]},
    )

    assert ops == {"demo": [impl_header]}
