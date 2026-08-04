import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest


def _load_resolver_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "resolve_linked_ops.py"
    )
    spec = importlib.util.spec_from_file_location("resolve_linked_ops_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_linked_config(root, *, library_extra="", binding_extra=""):
    platform = root / "cuda" / "metax"
    op_dir = platform / "ops" / "silu_and_mul"
    op_dir.mkdir(parents=True)
    (platform / "vllm.yaml").write_text(
        "transport: torch\n"
        "python_distribution: vllm\n"
        "library_glob: vllm/_C*.so\n"
        f"{library_extra}"
    )
    (op_dir / "binding.yaml").write_text(
        "library: vllm\n"
        "required_symbols:\n"
        "  - silu_and_mul(at::Tensor&, at::Tensor&)\n"
        f"{binding_extra}"
    )
    (op_dir / "adapter.cc").write_text("// adapter\n")
    return platform, op_dir


def test_resolve_collects_selected_adapter_and_library(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    _, op_dir = _write_linked_config(source_root)
    ignored_dir = source_root / "cuda" / "metax" / "ops" / "ignored"
    ignored_dir.mkdir()
    (ignored_dir / "binding.yaml").write_text(
        "library: missing\nrequired_symbols:\n  - missing()\n"
    )
    (ignored_dir / "adapter.cc").write_text("// ignored\n")
    library_path = tmp_path / "site-packages" / "vllm" / "_C.abi3.so"
    library_path.parent.mkdir(parents=True)
    library_path.touch()

    monkeypatch.setattr(
        module, "_locate_distribution_library", lambda config: library_path
    )
    exported = {"silu_and_mul(at::Tensor&, at::Tensor&)"}
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda library, nm, readelf: (exported, exported),
    )

    output_dir = tmp_path / "generated" / "linked"
    payload = module.resolve_linked_ops(
        ["cpu;metax"],
        ["silu_and_mul"],
        source_root=source_root,
        output_dir=output_dir,
    )

    assert payload["devices"] == ["cpu", "metax"]
    assert payload["operators"] == [
        {
            "adapter": str((op_dir / "adapter.cc").resolve()),
            "device": "metax",
            "family": "cuda",
            "library": "vllm",
            "name": "silu_and_mul",
            "required_symbols": ["silu_and_mul(at::Tensor&, at::Tensor&)"],
        }
    ]
    assert payload["libraries"][0]["path"] == str(library_path)
    assert json.loads((output_dir / "resolved.json").read_text()) == payload

    manifest = (output_dir / "manifest.cmake").read_text()
    assert "INFINI_OPS_LINKED_SOURCES" in manifest
    assert str(op_dir / "adapter.cc").replace("\\", "/") in manifest
    assert str(library_path).replace("\\", "/") in manifest
    assert '"torch"' in manifest
    assert "ignored" not in manifest


@pytest.mark.parametrize(
    ("library_extra", "binding_extra", "unknown_key"),
    (
        ("schema: silu_and_mul\n", "", "schema"),
        ("", "call: vllm::silu_and_mul\n", "call"),
    ),
)
def test_resolve_rejects_unknown_yaml_keys(
    tmp_path, library_extra, binding_extra, unknown_key
):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    _write_linked_config(
        source_root,
        library_extra=library_extra,
        binding_extra=binding_extra,
    )

    with pytest.raises(module.ResolutionError, match=f"unknown keys: {unknown_key}"):
        module.resolve_linked_ops(
            ["metax"],
            source_root=source_root,
            output_dir=tmp_path / "generated",
        )


def test_resolve_rejects_symbol_missing_from_either_tool(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    _write_linked_config(source_root)
    library_path = tmp_path / "_C.so"
    library_path.touch()

    monkeypatch.setattr(
        module, "_locate_distribution_library", lambda config: library_path
    )
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda library, nm, readelf: (
            {"silu_and_mul(at::Tensor&, at::Tensor&)"},
            set(),
        ),
    )

    with pytest.raises(module.ResolutionError, match="according to readelf"):
        module.resolve_linked_ops(
            ["metax"],
            source_root=source_root,
            output_dir=tmp_path / "generated",
        )


def test_resolve_requires_exact_demangled_symbol(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    _write_linked_config(source_root)
    library_path = tmp_path / "_C.so"
    library_path.touch()

    monkeypatch.setattr(
        module, "_locate_distribution_library", lambda config: library_path
    )
    exported = {"vllm::silu_and_mul(at::Tensor&, at::Tensor&)"}
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda library, nm, readelf: (exported, exported),
    )

    with pytest.raises(module.ResolutionError, match="according to nm and readelf"):
        module.resolve_linked_ops(
            ["metax"],
            source_root=source_root,
            output_dir=tmp_path / "generated",
        )


def test_resolve_rejects_distinct_libraries_with_same_basename(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    platform, _ = _write_linked_config(source_root)
    (platform / "other.yaml").write_text(
        "transport: torch\npython_distribution: other\nlibrary_glob: other/_C*.so\n"
    )
    other_op_dir = platform / "ops" / "other_op"
    other_op_dir.mkdir()
    (other_op_dir / "binding.yaml").write_text(
        "library: other\nrequired_symbols:\n  - other_op()\n"
    )
    (other_op_dir / "adapter.cc").write_text("// adapter\n")

    vllm_library = tmp_path / "vllm" / "_C.abi3.so"
    other_library = tmp_path / "other" / "_C.abi3.so"
    vllm_library.parent.mkdir()
    other_library.parent.mkdir()
    vllm_library.touch()
    other_library.touch()
    libraries = {"vllm": vllm_library, "other": other_library}
    monkeypatch.setattr(
        module,
        "_locate_distribution_library",
        lambda config: libraries[config.name],
    )
    exports = {
        vllm_library: {"silu_and_mul(at::Tensor&, at::Tensor&)"},
        other_library: {"other_op()"},
    }
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda library, nm, readelf: (exports[library], exports[library]),
    )
    output_dir = tmp_path / "generated"

    with pytest.raises(module.ResolutionError, match="share basename '_C.abi3.so'"):
        module.resolve_linked_ops(
            ["metax"], source_root=source_root, output_dir=output_dir
        )
    assert not (output_dir / "manifest.cmake").exists()


def test_resolve_rejects_symbol_exported_by_distinct_libraries(monkeypatch, tmp_path):
    module = _load_resolver_module()
    source_root = tmp_path / "linked"
    platform, _ = _write_linked_config(source_root)
    (platform / "other.yaml").write_text(
        "transport: torch\npython_distribution: other\nlibrary_glob: other/_C*.so\n"
    )
    other_op_dir = platform / "ops" / "other_op"
    other_op_dir.mkdir()
    (other_op_dir / "binding.yaml").write_text(
        "library: other\nrequired_symbols:\n  - other_op()\n"
    )
    (other_op_dir / "adapter.cc").write_text("// adapter\n")

    vllm_library = tmp_path / "vllm" / "_C.vllm.so"
    other_library = tmp_path / "other" / "_C.other.so"
    vllm_library.parent.mkdir()
    other_library.parent.mkdir()
    vllm_library.touch()
    other_library.touch()
    libraries = {"vllm": vllm_library, "other": other_library}
    monkeypatch.setattr(
        module,
        "_locate_distribution_library",
        lambda config: libraries[config.name],
    )
    exports = {
        vllm_library: {"silu_and_mul(at::Tensor&, at::Tensor&)"},
        other_library: {
            "silu_and_mul(at::Tensor&, at::Tensor&)",
            "other_op()",
        },
    }
    monkeypatch.setattr(
        module,
        "_inspect_dynamic_symbols",
        lambda library, nm, readelf: (exports[library], exports[library]),
    )

    with pytest.raises(module.ResolutionError, match="exported by multiple"):
        module.resolve_linked_ops(
            ["metax"],
            source_root=source_root,
            output_dir=tmp_path / "generated",
        )


def test_dynamic_symbol_inspection_uses_nm_and_readelf(monkeypatch, tmp_path):
    module = _load_resolver_module()
    library_path = tmp_path / "_C.so"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "llvm-nm":
            output = "0000000000001234 T vllm::silu_and_mul(at::Tensor&, at::Tensor&)\n"
        else:
            output = (
                "  12: 0000000000001234 42 FUNC GLOBAL DEFAULT 12 "
                "vllm::silu_and_mul(at::Tensor&, at::Tensor&)\n"
            )
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    nm_symbols, readelf_symbols = module._inspect_dynamic_symbols(
        library_path, "llvm-nm", "llvm-readelf"
    )

    expected = {"vllm::silu_and_mul(at::Tensor&, at::Tensor&)"}
    assert nm_symbols == expected
    assert readelf_symbols == expected
    assert [call[0] for call in calls] == [
        ["llvm-nm", "-D", "--defined-only", "-C", str(library_path)],
        [
            "llvm-readelf",
            "--dyn-syms",
            "--wide",
            "--demangle",
            str(library_path),
        ],
    ]
    assert all(
        kwargs == {"check": True, "capture_output": True, "text": True}
        for _, kwargs in calls
    )


def test_library_glob_matches_path_segments():
    module = _load_resolver_module()

    assert module._matches_library_glob("vllm/_C.abi3.so", "vllm/_C*.so")
    assert not module._matches_library_glob("vllm/nested/_C.abi3.so", "vllm/_C*.so")


def test_locate_distribution_library_requires_one_glob_match(monkeypatch, tmp_path):
    module = _load_resolver_module()
    site_packages = tmp_path / "site-packages"
    library = site_packages / "vllm" / "_C.abi3.so"
    library.parent.mkdir(parents=True)
    library.touch()
    outside_library = tmp_path / "outside" / "_C.evil.so"
    outside_library.parent.mkdir()
    outside_library.touch()

    class FakeDistribution:
        files = (
            pathlib.PurePosixPath("vllm/__init__.py"),
            pathlib.PurePosixPath("vllm/_C.abi3.so"),
            pathlib.PurePosixPath("vllm/_C.evil.so"),
        )

        def locate_file(self, entry):
            if str(entry) == "vllm/_C.evil.so":
                return outside_library
            return site_packages / entry

    monkeypatch.setattr(
        module.importlib.metadata, "distribution", lambda name: FakeDistribution()
    )
    config = module.LibraryConfig(
        device="metax",
        family="cuda",
        name="vllm",
        path=tmp_path / "vllm.yaml",
        transport="torch",
        python_distribution="vllm",
        library_glob="vllm/_C*.so",
    )

    assert module._locate_distribution_library(config) == library.resolve()


def test_locate_distribution_library_falls_back_to_distribution_root(
    monkeypatch, tmp_path
):
    module = _load_resolver_module()
    site_packages = tmp_path / "site-packages"
    library = site_packages / "vllm" / "_C.abi3.so"
    library.parent.mkdir(parents=True)
    library.touch()

    class FakeDistribution:
        files = None

        def locate_file(self, entry):
            return site_packages / entry

    monkeypatch.setattr(
        module.importlib.metadata, "distribution", lambda name: FakeDistribution()
    )
    config = module.LibraryConfig(
        device="moore",
        family="musa",
        name="vllm",
        path=tmp_path / "vllm.yaml",
        transport="torch",
        python_distribution="vllm",
        library_glob="vllm/_C*.so",
    )

    assert module._locate_distribution_library(config) == library.resolve()
