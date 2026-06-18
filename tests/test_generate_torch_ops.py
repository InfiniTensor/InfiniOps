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

    assert module._public_op_name("_softmax") == "internal_softmax"
    assert module._public_op_name("add_") == "add"
    assert module._public_op_name("_add_relu_") == "internal_add_relu"


def test_schema_self_param_renders_as_input_in_public_cpp_api():
    module = _load_generator_module()
    op = module._parse_func(
        "_softmax(Tensor self, int dim, bool half_to_float, *, "
        "Tensor(a!) out) -> Tensor(a!)"
    )

    assert op.params[0].name == "self"
    assert op.params[0].api_name == "input"

    base = module._generate_base_header("internal_softmax", [op])
    source = module._generate_torch_method_source("internal_softmax", op)

    assert "Softmax(const Tensor input, const int64_t dim" in base
    assert "self_shape_" not in base
    assert "input_shape_" in base
    assert "auto at_self = ToAtenTensor<kDev>" in source
    assert "input_shape_" in source
    assert "at::_softmax_out(at_out, at_self" in source


def test_optional_tensor_params_are_exposed_and_forwarded_to_aten():
    module = _load_generator_module()
    op = module._parse_func(
        "batch_norm_elemt(Tensor input, Tensor? weight=None, "
        "Tensor? bias=None, Tensor mean, Tensor invstd, float eps, "
        "*, Tensor(a!) out) -> Tensor(a!)"
    )

    assert [param.cpp_type for param in op.visible_params] == [
        "Tensor",
        "std::optional<Tensor>",
        "std::optional<Tensor>",
        "Tensor",
        "Tensor",
        "double",
        "Tensor",
    ]

    base = module._generate_base_header("batch_norm_elemt", [op])
    source = module._generate_torch_method_source("batch_norm_elemt", op)

    assert "#include <optional>" in base
    assert "std::optional<Tensor> weight" in base
    assert "std::optional<Tensor> bias" in base
    assert "bool has_weight_" in base
    assert "bool has_bias_" in base
    assert "c10::optional<at::Tensor> at_weight" in source
    assert "c10::optional<at::Tensor> at_bias" in source
    assert "weight->shape()" in source
    assert "weight_shape_" not in source
    assert "at::batch_norm_elemt_out" in source
    assert "at_weight" in source
    assert "at_bias" in source


def test_tensor_list_params_are_exposed_and_forwarded_to_aten():
    module = _load_generator_module()
    op = module._parse_func(
        "stack(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)"
    )

    assert [param.cpp_type for param in op.visible_params] == [
        "std::vector<Tensor>",
        "int64_t",
        "Tensor",
    ]
    assert op.is_testable

    base = module._generate_base_header("stack", [op])
    source = module._generate_torch_method_source("stack", op)

    assert "#include <vector>" in base
    assert "std::vector<Tensor> tensors" in base
    assert "std::vector<at::Tensor> at_tensors" in source
    assert "at_tensors.reserve(tensors.size())" in source
    assert "for (const auto& tensor : tensors)" in source
    assert "at::stack_out(at_out, at_tensors, dim)" in source


def test_optional_scalar_and_array_params_are_exposed_and_forwarded_to_aten():
    module = _load_generator_module()
    quantile = module._parse_func(
        "quantile(Tensor input, Tensor q, int? dim=None, bool keepdim=False, "
        "str interpolation='linear', *, Tensor(a!) out) -> Tensor(a!)"
    )
    upsample = module._parse_func(
        "upsample_bicubic2d(Tensor input, SymInt[2] output_size, "
        "bool align_corners, float[]? scale_factors=None, "
        "*, Tensor(a!) out) -> Tensor(a!)"
    )

    assert [param.cpp_type for param in quantile.visible_params] == [
        "Tensor",
        "Tensor",
        "std::optional<int64_t>",
        "bool",
        "std::string",
        "Tensor",
    ]
    assert [param.cpp_type for param in upsample.visible_params] == [
        "Tensor",
        "std::vector<int64_t>",
        "bool",
        "std::optional<std::vector<double>>",
        "Tensor",
    ]

    quantile_source = module._generate_torch_method_source("quantile", quantile)
    upsample_source = module._generate_torch_method_source(
        "upsample_bicubic2d", upsample
    )

    assert "c10::optional<int64_t> at_dim" in quantile_source
    assert "at::quantile_out" in quantile_source
    assert "at_dim" in quantile_source
    assert "c10::optional<at::ArrayRef<double>> at_scale_factors" in upsample_source
    assert "at::upsample_bicubic2d_out" in upsample_source
    assert "at_scale_factors" in upsample_source


def test_required_scalar_type_params_use_public_data_type():
    module = _load_generator_module()
    op = module._parse_func(
        "_softmax_backward_data(Tensor grad_output, Tensor output, int dim, "
        "ScalarType input_dtype, *, Tensor(a!) grad_input) -> Tensor(a!)"
    )

    assert [param.cpp_type for param in op.visible_params] == [
        "Tensor",
        "Tensor",
        "int64_t",
        "DataType",
        "Tensor",
    ]

    source = module._generate_torch_method_source("internal_softmax_backward_data", op)

    assert "at::_softmax_backward_data_out" in source
    assert "ToAtenDataType(input_dtype)" in source


def test_existing_base_overload_can_omit_optional_schema_params():
    module = _load_generator_module()
    op = module._parse_func(
        "slow_conv3d(Tensor input, Tensor weight, int[3] kernel_size, "
        "Tensor? bias=None, int[3] stride=1, int[3] padding=0, "
        "*, Tensor(a!) out) -> Tensor(a!)"
    )
    signature = [
        ("Tensor", "input"),
        ("Tensor", "weight"),
        ("std::vector<int64_t>", "kernel_size"),
        ("std::vector<int64_t>", "stride"),
        ("std::vector<int64_t>", "padding"),
        ("Tensor", "out"),
    ]

    bound = module._bind_base_signature(op, signature)

    assert bound is not None
    assert [param.name for param in bound.visible_params] == [
        "input",
        "weight",
        "kernel_size",
        "stride",
        "padding",
        "out",
    ]

    source = module._generate_torch_method_source("slow_conv3d", bound)

    assert "std::optional<Tensor> bias" not in source
    assert "c10::optional<at::Tensor>{}" in source
    assert "at::slow_conv3d_out" in source


def test_existing_base_overload_can_omit_defaulted_schema_params():
    module = _load_generator_module()
    op = module._parse_func(
        "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1, "
        "Tensor(a!) out) -> Tensor(a!)"
    )
    signature = [
        ("const Tensor", "input"),
        ("const Tensor", "other"),
        ("Tensor", "out"),
    ]

    bound = module._bind_base_signature(op, signature)

    assert bound is not None

    source = module._generate_torch_method_source("add", bound)

    assert "double alpha" not in source
    assert "const auto device_index = out.device().index();" in source
    assert "device_index_)" not in source
    assert "at::add_out(at_out, at_self, at_other, 1)" in source


def test_existing_base_overload_matches_by_name_when_types_repeat():
    module = _load_generator_module()
    op = module._parse_func(
        "std(Tensor input, int[1]? dim=None, bool unbiased=True, "
        "bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"
    )
    signature = [
        ("Tensor", "input"),
        ("bool", "keepdim"),
        ("Tensor", "out"),
    ]

    bound = module._bind_base_signature(op, signature)

    assert bound is not None
    assert [param.name for param in bound.visible_params] == [
        "input",
        "keepdim",
        "out",
    ]

    source = module._generate_torch_method_source("std", bound)

    assert "c10::optional<at::IntArrayRef>{}, true, keepdim" in source
    assert "unbiased" not in source


def test_write_text_if_changed_preserves_unchanged_mtime(tmp_path):
    module = _load_generator_module()
    path = tmp_path / "generated.cc"
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
    keep = root / "torch" / "keep.cc"
    stale = root / "torch" / "stale.cc"
    nested_stale = root / "base" / "stale.h"
    keep.parent.mkdir(parents=True)
    nested_stale.parent.mkdir(parents=True)
    keep.write_text("keep\n")
    stale.write_text("stale\n")
    nested_stale.write_text("stale\n")

    module._remove_stale_files(root, {keep})

    assert keep.exists()
    assert not stale.exists()
    assert not nested_stale.exists()
    assert not (root / "base").exists()
