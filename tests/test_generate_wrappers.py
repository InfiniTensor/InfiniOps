import importlib.util
import pathlib
import sys


def _load_wrappers_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_wrappers.py"
    )
    spec = importlib.util.spec_from_file_location("generate_wrappers_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def test_type_spelling_helpers_distinguish_optional_tensor_from_scalar():
    module = _load_wrappers_module()

    assert module._spelling_is_optional_tensor("const std::optional<Tensor>")
    assert not module._spelling_is_optional_tensor("const std::optional<double>")
    assert module._spelling_is_top_level_optional("const std::optional<double>")
    assert not module._spelling_is_top_level_optional(
        "const std::vector<std::optional<Tensor>>"
    )
    assert module._spelling_is_vector_tensor("const std::vector<Tensor>")
    assert not module._spelling_is_vector_tensor("const std::vector<int64_t>")
    assert not module._spelling_is_vector_tensor(
        "const std::vector<std::optional<Tensor>>"
    )
    assert module._spelling_is_vector_optional_tensor(
        "const std::vector<std::optional<Tensor>>"
    )
    assert not module._spelling_is_vector_optional_tensor(
        "const std::vector<Tensor>"
    )
    assert module._spelling_is_vector_int64("const std::vector<int64_t>")
    assert not module._spelling_is_vector_int64("const std::vector<Tensor>")


def test_generate_pybind11_uses_vector_optional_tensor_binding(tmp_path, monkeypatch):
    module = _load_wrappers_module()

    header = tmp_path / "index.h"
    header.write_text(
        """class Index {
 public:
  Index(const Tensor input, const std::vector<std::optional<Tensor>> indices, Tensor out);
  void operator()(const Tensor input, const std::vector<std::optional<Tensor>> indices, Tensor out) const;
};
"""
    )

    monkeypatch.setattr(module, "_find_base_header", lambda _op_name: header)

    class _FakeType:
        def __init__(self, spelling):
            self.spelling = spelling

    class _FakeArg:
        def __init__(self, spelling, type_spelling):
            self.spelling = spelling
            self.type = _FakeType(type_spelling)

    class _FakeNode:
        def __init__(self, *args):
            self._args = args

        def get_arguments(self):
            return self._args

    class _FakeOperator:
        def __init__(self):
            args = (
                _FakeArg("input", "const Tensor"),
                _FakeArg("indices", "const std::vector<std::optional<Tensor>>"),
                _FakeArg("out", "Tensor"),
            )
            self.name = "index"
            self.constructors = (_FakeNode(*args),)
            self.calls = (_FakeNode(*args),)

    source = module._generate_pybind11(_FakeOperator())

    assert "const std::vector<py::object> indices" in source
    assert "VectorOptionalTensorFromPybind11Handle(indices)" in source
    assert 'py::arg("indices") = py::none()' not in source
