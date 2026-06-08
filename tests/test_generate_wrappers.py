import importlib.util
import pathlib
import sys


def _load_wrappers_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / "generate_wrappers.py"
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
    assert not module._spelling_is_vector_optional_tensor("const std::vector<Tensor>")
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


def test_generate_dispatch_preserves_optional_scalar_and_tensor_overloads(
    tmp_path, monkeypatch
):
    module = _load_wrappers_module()

    header = tmp_path / "clamp.h"
    header.write_text(
        """class Clamp {
 public:
  Clamp(const Tensor input, const std::optional<double> min, const std::optional<double> max, Tensor out);
  Clamp(const Tensor input, const std::optional<Tensor> min, const std::optional<Tensor> max, Tensor out);
  void operator()(const Tensor input, const std::optional<double> min, const std::optional<double> max, Tensor out) const;
  void operator()(const Tensor input, const std::optional<Tensor> min, const std::optional<Tensor> max, Tensor out) const;
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

    scalar_args = (
        _FakeArg("input", "const Tensor"),
        _FakeArg("min", "const std::optional<double>"),
        _FakeArg("max", "const std::optional<double>"),
        _FakeArg("out", "Tensor"),
    )
    tensor_args = (
        _FakeArg("input", "const Tensor"),
        _FakeArg("min", "const std::optional<Tensor>"),
        _FakeArg("max", "const std::optional<Tensor>"),
        _FakeArg("out", "Tensor"),
    )

    class _FakeOperator:
        name = "clamp"
        constructors = (_FakeNode(*scalar_args), _FakeNode(*tensor_args))
        calls = (_FakeNode(*scalar_args), _FakeNode(*tensor_args))

    declarations, _ = module._generate_generated_dispatch_entries(_FakeOperator())
    source = "\n".join(declarations)

    assert "const std::optional<double> min" in source
    assert "const std::optional<double> max" in source
    assert "std::optional<Tensor> min" in source
    assert "std::optional<Tensor> max" in source


def test_generate_dispatch_preserves_vector_optional_tensor(tmp_path, monkeypatch):
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

    args = (
        _FakeArg("input", "const Tensor"),
        _FakeArg("indices", "const std::vector<std::optional<Tensor>>"),
        _FakeArg("out", "Tensor"),
    )

    class _FakeOperator:
        name = "index"
        constructors = (_FakeNode(*args),)
        calls = (_FakeNode(*args),)

    declarations, _ = module._generate_generated_dispatch_entries(_FakeOperator())
    source = "\n".join(declarations)

    assert "std::vector<std::optional<Tensor>> indices" in source
    assert "std::optional<Tensor> indices" not in source


def test_write_text_if_changed_preserves_unchanged_mtime(tmp_path):
    module = _load_wrappers_module()
    path = tmp_path / "bindings.cc"
    path.write_text("same\n")
    before = path.stat().st_mtime_ns

    assert module._write_text_if_changed(path, "same\n") is False
    assert path.stat().st_mtime_ns == before

    assert module._write_text_if_changed(path, "different\n") is True
    assert path.read_text() == "different\n"
    assert path.stat().st_mtime_ns >= before


def test_remove_stale_files_keeps_expected_outputs(tmp_path):
    module = _load_wrappers_module()
    root = tmp_path / "generated"
    keep = root / "bindings" / "keep.cc"
    stale = root / "bindings" / "stale.cc"
    nested_stale = root / "src" / "foo" / "operator.cc"
    keep.parent.mkdir(parents=True)
    nested_stale.parent.mkdir(parents=True)
    keep.write_text("keep\n")
    stale.write_text("stale\n")
    nested_stale.write_text("stale\n")

    module._remove_stale_files(root, {keep})

    assert keep.exists()
    assert not stale.exists()
    assert not nested_stale.exists()
    assert not (root / "src" / "foo").exists()
