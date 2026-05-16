import importlib.util
import pathlib
import sys


def _load_generator_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_torch_ops.py"
    )
    spec = importlib.util.spec_from_file_location("generate_torch_ops_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def test_load_aten_yaml_uses_packaged_torchgen(monkeypatch):
    module = _load_generator_module()
    monkeypatch.setattr(module, "_load_packaged_aten_yaml", lambda: "packaged: true\n")

    assert module._load_aten_yaml("v9.9.9") == "packaged: true\n"


def test_public_op_name_normalizes_aten_internal_and_inplace_names():
    module = _load_generator_module()

    assert module._public_op_name("_softmax") == "aten_softmax"
    assert module._public_op_name("add_") == "add_inplace"
    assert module._public_op_name("_add_relu_") == "aten_add_relu_inplace"
