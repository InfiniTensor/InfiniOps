import importlib.util
import json
import pathlib
import subprocess
import sys


def _load_matrix_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "infini_ops_plugin_test_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("infini_ops_plugin_test_matrix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def _plugin_root():
    return pathlib.Path(__file__).resolve().parents[1] / "plugins"


def test_device_source_change_maps_to_one_platform():
    module = _load_matrix_module()

    matrix = module.build_test_matrix(
        _plugin_root(), ["src/native/cuda/nvidia/ops/add/add.cu"]
    )

    assert matrix["plugins"] == ["nvidia"]
    assert matrix["devices"] == ["nvidia"]
    assert matrix["test_devices"] == ["cuda"]
    assert matrix["ci_platforms"] == ["nvidia"]
    assert matrix["requires_full_matrix"] is False


def test_shared_plugin_change_maps_to_dependents():
    module = _load_matrix_module()

    matrix = module.build_test_matrix(
        _plugin_root(), ["src/native/cuda/ops/gemm/gemm.cc"]
    )

    assert matrix["plugins"] == [
        "cuda-common",
        "hygon",
        "iluvatar",
        "metax",
        "moore",
        "nvidia",
    ]
    assert matrix["devices"] == ["hygon", "iluvatar", "metax", "moore", "nvidia"]
    assert matrix["test_devices"] == ["cuda", "musa"]
    assert matrix["ci_platforms"] == ["hygon", "iluvatar", "metax", "moore", "nvidia"]
    assert matrix["requires_full_matrix"] is False


def test_core_codegen_change_requests_full_matrix():
    module = _load_matrix_module()

    matrix = module.build_test_matrix(
        _plugin_root(), ["scripts/generate_wrappers.py"]
    )

    assert matrix["devices"] == [
        "ascend",
        "cambricon",
        "cpu",
        "hygon",
        "iluvatar",
        "metax",
        "moore",
        "nvidia",
    ]
    assert matrix["ci_platforms"] == [
        "ascend",
        "cambricon",
        "hygon",
        "iluvatar",
        "metax",
        "moore",
        "nvidia",
    ]
    assert matrix["requires_full_matrix"] is True


def test_plugin_manifest_change_maps_to_that_plugin():
    module = _load_matrix_module()

    matrix = module.build_test_matrix(_plugin_root(), ["plugins/cambricon/plugin.json"])

    assert matrix["plugins"] == ["cambricon"]
    assert matrix["devices"] == ["cambricon"]
    assert matrix["test_devices"] == ["mlu"]
    assert matrix["ci_platforms"] == ["cambricon"]


def test_matrix_accepts_multiple_plugin_roots(tmp_path):
    module = _load_matrix_module()
    external_root = tmp_path / "external"
    plugin_dir = external_root / "external-cpu"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.cmake").write_text("# external plugin\n", encoding="utf-8")
    (plugin_dir / "plugin.json").write_text(
        json.dumps(
            {
                "name": "external-cpu",
                "kind": "device",
                "contract_version": 1,
                "devices": ["cpu"],
                "depends": [],
                "cmake_entry": "plugin.cmake",
                "source_roots": ["external/native/cpu"],
                "operator_roots": ["external/native/cpu/ops"],
                "device_headers": {"cpu": "external/native/cpu/device_.h"},
                "test_devices": {"cpu": "cpu"},
            }
        ),
        encoding="utf-8",
    )

    matrix = module.build_test_matrix(
        [_plugin_root(), external_root], ["external/native/cpu/ops/add.cc"]
    )

    assert matrix["plugins"] == ["external-cpu"]
    assert matrix["devices"] == ["cpu"]
    assert matrix["ci_platforms"] == []


def test_cli_outputs_json_matrix(tmp_path):
    repo = pathlib.Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "infini_ops_plugin_test_matrix.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--plugin-root",
            str(_plugin_root()),
            "plugins/cpu/plugin.cmake",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    matrix = json.loads(result.stdout)
    assert matrix["plugins"] == ["cpu"]
    assert matrix["devices"] == ["cpu"]
    assert matrix["ci_platforms"] == []
