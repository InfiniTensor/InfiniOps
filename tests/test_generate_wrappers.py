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
        "MakeClamp(const Config& config, Tensor input, "
        "const std::optional<double> min, const std::optional<double> max, "
        "Tensor out)"
    ) in text
    assert (
        "MakeClamp(const Config& config, Tensor input, "
        "std::optional<Tensor> min, std::optional<Tensor> max, Tensor out)"
    ) in text


def test_operator_call_instantiations_keep_optional_scalar_and_tensor_overloads_distinct(
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
        constructors=[],
        calls=[
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
    )

    declarations, _ = module._generate_operator_call_instantiation_entries(operator)

    text = "\n".join(declarations)

    assert (
        "Call<Tensor, std::optional<double>, std::optional<double>, Tensor>"
    ) in text
    assert (
        "Call<Tensor, std::optional<Tensor>, std::optional<Tensor>, Tensor>"
    ) in text


def test_extractor_prefers_header_types_for_reused_parameter_names(
    monkeypatch, tmp_path
):
    module = _load_generator_module()
    base_header = tmp_path / "histogram.h"
    base_header.write_text(
        """
class Histogram {
 public:
  virtual void operator()(const Tensor input, const Tensor bins,
                          const std::optional<Tensor> weight,
                          const bool density, Tensor hist,
                          Tensor bin_edges) const = 0;
  virtual void operator()(const Tensor input, const int64_t bins,
                          const std::optional<std::vector<double>> range,
                          const std::optional<Tensor> weight,
                          const bool density, Tensor hist,
                          Tensor bin_edges) const = 0;
};
"""
    )
    monkeypatch.setattr(module, "_find_base_header", lambda op_name: base_header)

    clang_calls = [
        module._ParsedFunction(
            [
                module._ParsedArgument("const int", "input"),
                module._ParsedArgument("const int", "bins"),
                module._ParsedArgument("const int", "weight"),
                module._ParsedArgument("const bool", "density"),
                module._ParsedArgument("int", "hist"),
                module._ParsedArgument("int", "bin_edges"),
            ]
        ),
        module._ParsedFunction(
            [
                module._ParsedArgument("const int", "input"),
                module._ParsedArgument("const int64_t", "bins"),
                module._ParsedArgument("const int", "range"),
                module._ParsedArgument("const int", "weight"),
                module._ParsedArgument("const bool", "density"),
                module._ParsedArgument("int", "hist"),
                module._ParsedArgument("int", "bin_edges"),
            ]
        ),
    ]

    operator = module._Operator(
        "histogram",
        constructors=[],
        calls=module._prefer_header_type_spellings(
            clang_calls, module._parse_operator_header("histogram").calls
        ),
    )

    declarations, _ = module._generate_operator_call_instantiation_entries(operator)
    text = "\n".join(declarations)

    assert (
        "Call<Tensor, int64_t, std::optional<std::vector<double>>, "
        "std::optional<Tensor>, bool, Tensor, Tensor>"
    ) in text
    assert "Call<Tensor, Tensor, int, std::optional<Tensor>" not in text


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


def test_write_text_if_changed_preserves_unchanged_mtime(tmp_path):
    module = _load_generator_module()
    path = tmp_path / "bindings.cc"
    path.write_text("same\n")
    before = path.stat().st_mtime_ns

    assert module._write_text_if_changed(path, "same\n") is False
    assert path.stat().st_mtime_ns == before

    assert module._write_text_if_changed(path, "different\n") is True
    assert path.read_text() == "different\n"
    assert path.stat().st_mtime_ns >= before


def test_remove_stale_files_keeps_expected_outputs(tmp_path):
    module = _load_generator_module()
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


def _generate_binding(op_name, tmp_path, monkeypatch, source):
    module = _load_generator_module()
    src_dir = tmp_path / "src"
    base_dir = src_dir / "base"
    base_dir.mkdir(parents=True)
    (base_dir / f"{op_name}.h").write_text(source)
    monkeypatch.setattr(module, "_SRC_DIR", src_dir)
    monkeypatch.setattr(module, "_BASE_DIR", base_dir)
    operator = module._OperatorExtractor()(op_name)

    return module._generate_pybind11(operator)


def test_mha_varlen_fwd_requires_out_binding(tmp_path, monkeypatch):
    text = _generate_binding(
        "mha_varlen_fwd",
        tmp_path,
        monkeypatch,
        """
#include <cstdint>
#include <optional>

namespace infini::ops {

struct Tensor {};

template <typename T>
class Operator {};

class MhaVarlenFwd : public Operator<MhaVarlenFwd> {
 public:
  MhaVarlenFwd(const Tensor q, const Tensor k, const Tensor v, Tensor out,
               const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
               std::optional<Tensor> block_table, float softmax_scale,
               bool is_causal, int64_t num_splits = 0) {}

  virtual void operator()(const Tensor q, const Tensor k, const Tensor v,
                          Tensor out, const Tensor cu_seqlens_q,
                          const Tensor cu_seqlens_k,
                          std::optional<Tensor> block_table,
                          float softmax_scale, bool is_causal,
                          int64_t num_splits = 0) const = 0;
};

}  // namespace infini::ops
""",
    )

    assert 'py::arg("out"), py::arg("cu_seqlens_q")' in text
    assert 'py::arg("num_splits") = 0' in text
    assert 'py::arg("out") = py::none()' not in text
    assert "std::optional<py::object> out" not in text
    assert "OptionalTensorFromPybind11Handle(out)" not in text


def test_mha_fwd_kvcache_requires_out_binding(tmp_path, monkeypatch):
    text = _generate_binding(
        "mha_fwd_kvcache",
        tmp_path,
        monkeypatch,
        """
#include <cstdint>
#include <optional>

namespace infini::ops {

struct Tensor {};

template <typename T>
class Operator {};

class MhaFwdKvcache : public Operator<MhaFwdKvcache> {
 public:
  MhaFwdKvcache(const Tensor q, const Tensor kcache, const Tensor vcache,
                std::optional<Tensor> k, std::optional<Tensor> v, Tensor out,
                float softmax_scale, bool is_causal,
                int64_t num_splits = 0) {}

  virtual void operator()(const Tensor q, const Tensor kcache,
                          const Tensor vcache, std::optional<Tensor> k,
                          std::optional<Tensor> v, Tensor out,
                          float softmax_scale, bool is_causal,
                          int64_t num_splits = 0) const = 0;
};

}  // namespace infini::ops
""",
    )

    assert 'py::arg("out"), py::arg("softmax_scale")' in text
    assert 'py::arg("num_splits") = 0' in text
    assert 'py::arg("out") = py::none()' not in text
    assert "std::optional<py::object> out" not in text
    assert "OptionalTensorFromPybind11Handle(out)" not in text
