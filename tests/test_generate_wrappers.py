import importlib.util
import pathlib
import sys


def _load_generator_module():
    path = (
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / "generate_wrappers.py"
    )
    spec = importlib.util.spec_from_file_location("generate_wrappers_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def test_generated_dispatch_keeps_optional_scalar_and_tensor_overloads_distinct(
    monkeypatch, tmp_path
):
    module = _load_generator_module()
    base_header = tmp_path / "clamp.h"
    base_header.write_text(
        """
class Clamp {
 public:
  virtual void operator()(const Tensor input, const std::optional<double> min,
                          const std::optional<double> max, Tensor out) const = 0;
  virtual void operator()(const Tensor input, const std::optional<Tensor> min,
                          const std::optional<Tensor> max, Tensor out) const = 0;
};
"""
    )
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)

    operator = module._Operator(
        "clamp",
        constructors=[
            module._ParsedFunction(
                [
                    module._ParsedArgument("const Tensor", "input"),
                    module._ParsedArgument("const std::optional<double>", "min"),
                    module._ParsedArgument("const std::optional<double>", "max"),
                    module._ParsedArgument("Tensor", "out"),
                ]
            ),
            module._ParsedFunction(
                [
                    module._ParsedArgument("const Tensor", "input"),
                    module._ParsedArgument("const std::optional<Tensor>", "min"),
                    module._ParsedArgument("const std::optional<Tensor>", "max"),
                    module._ParsedArgument("Tensor", "out"),
                ]
            ),
        ],
        calls=[],
    )

    declarations, _ = module._generate_generated_dispatch_entries(operator)

    text = "\n".join(declarations)

    assert (
        "MakeClamp(const Config& config, const Tensor input, "
        "const std::optional<double> min, const std::optional<double> max, "
        "Tensor out)"
    ) in text
    assert (
        "MakeClamp(const Config& config, const Tensor input, "
        "std::optional<Tensor> min, std::optional<Tensor> max, Tensor out)"
    ) in text


def test_pybind_default_implementation_uses_first_active_index(monkeypatch, tmp_path):
    module = _load_generator_module()
    base_header = tmp_path / "mul.h"
    base_header.write_text(
        """
class Mul {
 public:
  Mul(const Tensor input, const Tensor other, Tensor out);
  virtual void operator()(const Tensor input, const Tensor other, Tensor out) const = 0;
};
"""
    )
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)

    operator = module._Operator(
        "mul",
        constructors=[
            module._ParsedFunction(
                [
                    module._ParsedArgument("const Tensor", "input"),
                    module._ParsedArgument("const Tensor", "other"),
                    module._ParsedArgument("Tensor", "out"),
                ]
            )
        ],
        calls=[
            module._ParsedFunction(
                [
                    module._ParsedArgument("const Tensor", "input"),
                    module._ParsedArgument("const Tensor", "other"),
                    module._ParsedArgument("Tensor", "out"),
                ]
            )
        ],
    )

    text = module._generate_pybind11(operator)

    assert "std::size_t DefaultImplementationIndexForMul" in text
    assert (
        "config.set_implementation_index("
        "DefaultImplementationIndexForMul(DeviceFromPybind11Handle(input).type()))"
    ) in text
    assert "std::optional<std::size_t> implementation_index" in text
    assert (
        "implementation_index.value_or("
        "DefaultImplementationIndexForMul(DeviceFromPybind11Handle(input).type()))"
    ) in text
    assert 'py::arg("implementation_index") = py::none()' in text


def test_normalize_op_allowlist_accepts_spaces_and_commas():
    module = _load_generator_module()

    assert module._normalize_op_allowlist(["add,mul", " cast ", "", "gemm"]) == [
        "add",
        "mul",
        "cast",
        "gemm",
    ]


def test_filter_ops_preserves_allowlist_order_and_skips_unavailable_ops():
    module = _load_generator_module()
    ops = {"add": ["add_impl"], "mul": ["mul_impl"], "gemm": ["gemm_impl"]}

    assert module._filter_ops(ops, ["gemm", "add"]) == {
        "gemm": ["gemm_impl"],
        "add": ["add_impl"],
    }
    assert module._filter_ops(ops, ["add", "missing"]) == {"add": ["add_impl"]}


def test_filter_ops_strict_rejects_unavailable_ops():
    module = _load_generator_module()
    ops = {"add": ["add_impl"]}

    try:
        module._filter_ops(ops, ["add", "missing"], strict=True)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("strict unknown ops should fail")
