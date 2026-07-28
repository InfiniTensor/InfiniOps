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
    assert "auto self_lease = input_pool_.Acquire(input);" in source
    assert "auto& at_self = self_lease.Native();" in source
    assert "self_pool_" not in source
    assert "at::_softmax_out(at_out, at_self" in source


def test_abs_torch_backend_uses_generic_target_tensor_pools():
    module = _load_generator_module()
    op = module._parse_func("abs(Tensor self, *, Tensor(a!) out) -> Tensor(a!)")

    header = module._generate_torch_header("abs", [op])
    source = module._generate_torch_source("abs", [op])

    assert "using Abs::Abs;" in header
    assert "TargetTensorPool<AtenTensorAdapter<kDev>> input_pool_;" in header
    assert "TargetTensorPool<AtenTensorAdapter<kDev>> out_pool_;" in header
    assert "auto self_lease = input_pool_.Acquire(input);" in source
    assert "auto& at_self = self_lease.Native();" in source
    assert "auto out_lease = out_pool_.Acquire(out);" in source
    assert "auto& at_out = out_lease.Native();" in source
    assert "AtenTensorSlot" not in header
    assert "AtenTensorSlot" not in source
    assert "ToAtenTensor<kDev>" not in source


def test_pybind_tensor_metadata_bridge_does_not_retain_source_handles():
    root = pathlib.Path(__file__).resolve().parent.parent
    pybind_utils = (root / "src" / "pybind11_utils.h").read_text(encoding="utf-8")
    bridge_header = (root / "src" / "torch" / "pybind11_.h").read_text(encoding="utf-8")
    bridge_source = (root / "src" / "torch" / "pybind11_.cc").read_text(
        encoding="utf-8"
    )

    assert '"torch/pybind11_.h"' in pybind_utils
    assert "TryAtenTensorMetadataFromPyObject" in pybind_utils
    assert "if (metadata.has_value())" in pybind_utils
    assert "#ifdef WITH_TORCH" in pybind_utils
    assert 'obj.attr("data_ptr")' in pybind_utils
    assert "torch/csrc/" not in pybind_utils
    assert "torch/csrc/autograd/python_variable.h" not in bridge_header
    assert "torch/csrc/autograd/python_variable.h" in bridge_source
    assert "THPVariable_Check" in bridge_source
    assert "THPVariable_Unpack" in bridge_source
    assert "std::optional<AtenTensorMetadata>" in bridge_header
    assert "TryAtenTensorMetadataFromPyObject" in bridge_header
    assert "case at::kBool:" in bridge_source
    assert "throw " not in bridge_source
    assert "assert(" not in bridge_source
    assert "return std::nullopt;" in bridge_source
    assert "if (!THPVariable_Check(object))" in bridge_source
    assert "if (!dtype.has_value())" in bridge_source
    assert "device.has_index() ? device.index() : 0" in bridge_source
    assert "source_handle" not in bridge_header
    assert "source_handle" not in bridge_source
    assert "make_shared<at::Tensor>" not in bridge_source


def test_pybind_tensor_metadata_bridge_is_scoped_to_python_module():
    root = pathlib.Path(__file__).resolve().parent.parent
    cmake = (root / "src" / "CMakeLists.txt").read_text(encoding="utf-8")

    torch_section = cmake.split("if(WITH_TORCH)", maxsplit=1)[1]
    torch_section = torch_section.split("\nendif()", maxsplit=1)[0]

    assert "Development" not in torch_section
    assert "find_package(Python COMPONENTS Interpreter Development REQUIRED)" in cmake
    assert 'list(REMOVE_ITEM TORCH_SOURCES "${TORCH_PYBIND_BRIDGE_SOURCE}")' in cmake
    assert "add_library(infini_ops_torch_bridge_obj OBJECT" in cmake
    assert "-std=c++17 -fPIC -O2" in cmake
    assert '-MMD -MF "${_torch_bridge_dep}"' in cmake
    assert 'DEPFILE "${_torch_bridge_dep}"' in cmake
    assert '"${TORCH_PYBIND_BRIDGE_HEADER}"' in cmake
    assert "target_sources(ops PRIVATE" in cmake
    core_bridge = (
        "target_sources(infiniops PRIVATE\n"
        "            $<TARGET_OBJECTS:infini_ops_torch_bridge_obj>)"
    )
    assert core_bridge not in cmake
    assert "find_library(TORCH_PYTHON_LIB torch_python" in cmake
    assert "target_link_libraries(ops PRIVATE ${TORCH_PYTHON_LIB})" in cmake


def test_torch_header_orders_and_deduplicates_tensor_pools_across_overloads():
    module = _load_generator_module()
    tensor_overload = module._parse_func(
        "blend.Tensor(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)"
    )
    optional_overload = module._parse_func(
        "blend.optional(Tensor self, Tensor? weight=None, Tensor bias, *, "
        "Tensor(a!) out) -> Tensor(a!)"
    )
    list_overload = module._parse_func(
        "blend.list(Tensor self, Tensor[] other, Tensor[] tensors, *, "
        "Tensor(a!) out) -> Tensor(a!)"
    )

    header = module._generate_torch_header(
        "blend", [tensor_overload, optional_overload, list_overload]
    )
    members = [
        "TargetTensorPool<AtenTensorAdapter<kDev>> input_pool_;",
        "TargetTensorPool<AtenTensorAdapter<kDev>> other_pool_;",
        "TargetTensorPool<AtenTensorAdapter<kDev>> out_pool_;",
        "TargetTensorPool<AtenTensorAdapter<kDev>> weight_pool_;",
        "TargetTensorPool<AtenTensorAdapter<kDev>> bias_pool_;",
        "std::vector<TargetTensorPool<AtenTensorAdapter<kDev>>> other_pools_;",
        "std::vector<TargetTensorPool<AtenTensorAdapter<kDev>>> tensors_pools_;",
    ]

    assert [header.count(member) for member in members] == [1] * len(members)
    assert [header.index(member) for member in members] == sorted(
        header.index(member) for member in members
    )


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
    header = module._generate_torch_header("batch_norm_elemt", [op])
    source = module._generate_torch_method_source("batch_norm_elemt", op)

    assert "#include <optional>" in base
    assert "std::optional<Tensor> weight" in base
    assert "std::optional<Tensor> bias" in base
    assert "bool has_weight_" in base
    assert "bool has_bias_" in base
    assert "c10::optional<at::Tensor> at_weight" in source
    assert "c10::optional<at::Tensor> at_bias" in source
    assert (
        "std::optional<typename "
        "TargetTensorPool<AtenTensorAdapter<kDev>>::Lease> weight_lease;" in source
    )
    assert (
        "std::optional<typename "
        "TargetTensorPool<AtenTensorAdapter<kDev>>::Lease> bias_lease;" in source
    )
    assert "weight_lease.emplace(weight_pool_.Acquire(*weight));" in source
    assert "at_weight = weight_lease->Native();" in source
    assert "bias_lease.emplace(bias_pool_.Acquire(*bias));" in source
    assert "at_bias = bias_lease->Native();" in source
    assert source.index("weight_lease;") < source.index("at_weight;")
    assert source.index("bias_lease;") < source.index("at_bias;")
    assert source.index("at_weight;") < source.index("weight_lease.emplace")
    assert source.index("at_bias;") < source.index("bias_lease.emplace")
    assert source.index("at_weight = weight_lease->Native();") < source.index(
        "at::batch_norm_elemt_out"
    )
    assert source.index("at_bias = bias_lease->Native();") < source.index(
        "at::batch_norm_elemt_out"
    )
    assert "weight_pool_;" in header
    assert "bias_pool_;" in header
    assert "ToAtenTensor<kDev>" not in source
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
    header = module._generate_torch_header("stack", [op])
    source = module._generate_torch_method_source("stack", op)

    assert "#include <vector>" in base
    assert "std::vector<Tensor> tensors" in base
    list_pool = "std::vector<TargetTensorPool<AtenTensorAdapter<kDev>>> tensors_pools_;"
    lease_vector = (
        "std::vector<typename "
        "TargetTensorPool<AtenTensorAdapter<kDev>>::Lease> tensors_leases;"
    )
    assert list_pool in header
    assert header.count(list_pool) == 1
    assert "tensors_pool_;" not in header
    assert "if (tensors_pools_.size() < tensors.size()) {" in source
    assert "tensors_pools_.size() != tensors.size()" not in source
    assert "tensors_pools_.resize(tensors.size());" in source
    assert lease_vector in source
    assert "tensors_leases.reserve(tensors.size());" in source
    assert "std::vector<at::Tensor> at_tensors;" in source
    assert "at_tensors.reserve(tensors.size());" in source
    assert "for (std::size_t i = 0; i < tensors.size(); ++i)" in source
    assert "tensors_leases.push_back(tensors_pools_[i].Acquire(tensors[i]));" in source
    assert "at_tensors.push_back(tensors_leases.back().Native());" in source
    assert source.index("tensors_pools_.resize") < source.index(lease_vector)
    assert source.index("tensors_pools_.resize") < source.index(
        "tensors_pools_[i].Acquire"
    )
    assert source.index(lease_vector) < source.index(
        "std::vector<at::Tensor> at_tensors;"
    )
    assert source.index("std::vector<at::Tensor> at_tensors;") < source.index(
        "at::stack_out"
    )
    assert "ToAtenTensor<kDev>" not in source
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
    quantile_header = module._generate_torch_header("quantile", [quantile])
    upsample_header = module._generate_torch_header("upsample_bicubic2d", [upsample])

    assert "c10::optional<int64_t> at_dim" in quantile_source
    assert "at::quantile_out" in quantile_source
    assert "at_dim" in quantile_source
    assert "c10::optional<at::ArrayRef<double>> at_scale_factors" in upsample_source
    assert "at::upsample_bicubic2d_out" in upsample_source
    assert "at_scale_factors" in upsample_source
    assert "dim_pool_" not in quantile_header
    assert "scale_factors_pool_" not in upsample_header


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
    header = module._generate_torch_header("slow_conv3d", [bound])

    assert "std::optional<Tensor> bias" not in source
    assert "c10::optional<at::Tensor>{}" in source
    assert "bias_pool_" not in header
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
    assert "const auto device_index" not in source
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
