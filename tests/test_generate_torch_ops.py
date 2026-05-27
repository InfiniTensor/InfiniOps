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


def test_parse_func_marks_keyword_only_params():
    module = _load_generator_module()
    op = module._parse_func(
        "addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, "
        "Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"
    )
    params = {param.name: param for param in op.params}

    assert params["input"].keyword_only is False
    assert params["mat1"].keyword_only is False
    assert params["mat2"].keyword_only is False
    assert params["beta"].keyword_only is True
    assert params["alpha"].keyword_only is True
    assert params["out"].keyword_only is True


def test_metadata_marks_visible_params_after_hidden_slots_for_keyword_call():
    module = _load_generator_module()
    op = module._parse_func(
        "argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, "
        "Tensor(a!) out) -> Tensor(a!)"
    )
    params = module._metadata_param_dicts(op)

    assert params == [
        {
            "name": "input",
            "type": "Tensor",
            "is_tensor": True,
            "is_out": False,
            "keyword_only": False,
            "reference_keyword_only": False,
        },
        {
            "name": "keepdim",
            "type": "bool",
            "is_tensor": False,
            "is_out": False,
            "keyword_only": False,
            "reference_keyword_only": True,
        },
        {
            "name": "out",
            "type": "Tensor(a!)",
            "is_tensor": True,
            "is_out": True,
            "keyword_only": True,
            "reference_keyword_only": True,
        },
    ]


def test_parse_func_exposes_clamp_optional_bounds():
    module = _load_generator_module()
    op = module._parse_func(
        "clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, "
        "Tensor(a!) out) -> Tensor(a!)"
    )
    params = {param.name: param for param in op.params}

    assert params["min"].is_hidden is False
    assert params["max"].is_hidden is False
    assert params["min"].cpp_type == "std::optional<double>"
    assert params["max"].cpp_type == "std::optional<double>"


def test_parse_func_exposes_clip_tensor_bounds():
    module = _load_generator_module()
    op = module._parse_func(
        "clip.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, "
        "Tensor(a!) out) -> Tensor(a!)"
    )
    params = {param.name: param for param in op.params}

    assert params["min"].is_hidden is False
    assert params["max"].is_hidden is False
    assert params["min"].cpp_type == "std::optional<Tensor>"
    assert params["max"].cpp_type == "std::optional<Tensor>"


def test_generate_torch_method_source_converts_visible_optional_bounds():
    module = _load_generator_module()
    op = module._parse_func(
        "clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, "
        "Tensor(a!) out) -> Tensor(a!)"
    )

    source = module._generate_torch_method_source("clamp", op)

    assert "c10::optional<at::Scalar> at_min;" in source
    assert "c10::optional<at::Scalar> at_max;" in source
    assert "at_min = at::Scalar(min.value());" in source
    assert "at_max = at::Scalar(max.value());" in source
    assert "at::clamp_out(at_out, at_input, at_min, at_max)" in source


def test_parse_func_exposes_index_optional_tensor_list():
    module = _load_generator_module()
    op = module._parse_func(
        "index.Tensor_out(Tensor self, Tensor?[] indices, *, "
        "Tensor(a!) out) -> Tensor(a!)"
    )
    params = {param.name: param for param in op.params}

    assert params["indices"].is_hidden is False
    assert params["indices"].cpp_type == "std::vector<std::optional<Tensor>>"


def test_generate_torch_method_source_converts_visible_optional_tensor_list():
    module = _load_generator_module()
    op = module._parse_func(
        "index.Tensor_out(Tensor self, Tensor?[] indices, *, "
        "Tensor(a!) out) -> Tensor(a!)"
    )

    source = module._generate_torch_method_source("index", op)

    assert "c10::List<c10::optional<at::Tensor>> at_indices;" in source
    assert "at_indices.reserve(indices.size());" in source
    assert "at_indices.push_back(ToAtenTensor<kDev>(indices_tensor));" in source
    assert "at::index_out(at_out, at_input, at_indices)" in source


def test_generate_torch_method_source_prefers_functional_copy_for_selected_ops():
    module = _load_generator_module()
    op = module._parse_func(
        "fft_rfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, "
        "str? norm=None, *, Tensor(a!) out) -> Tensor(a!)"
    )

    source = module._generate_torch_method_source("fft_rfftn", op)

    assert "at_out.copy_(at::fft_rfftn(" in source
    assert "at::fft_rfftn_out(" not in source
