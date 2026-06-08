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


def test_generate_torch_method_source_prefers_functional_copy_for_multi_out_ops():
    module = _load_generator_module()
    op = module._parse_func(
        "mode.values(Tensor self, int dim=0, bool keepdim=False, *, "
        "Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)"
    )

    source = module._generate_torch_method_source("mode", op)

    assert "auto result = at::mode(" in source
    assert "at_values.copy_(std::get<0>(result))" in source
    assert "at_indices.copy_(std::get<1>(result))" in source
    assert "at::mode_out(" not in source


def test_generate_torch_method_source_uses_svd_for_svdvals_and_nuclear_norm():
    module = _load_generator_module()

    svdvals_op = module._parse_func(
        "linalg_svdvals.out(Tensor A, *, Tensor(a!) out) -> Tensor(a!)"
    )
    svdvals_source = module._generate_torch_method_source("linalg_svdvals", svdvals_op)
    assert "std::get<1>(at::linalg_svd(" in svdvals_source
    assert "compute_uv" not in svdvals_source

    nuclear_norm_op = module._parse_func(
        "nuclear_norm.out(Tensor self, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)"
    )
    nuclear_norm_source = module._generate_torch_method_source(
        "nuclear_norm", nuclear_norm_op
    )
    assert "std::get<1>(at::linalg_svd(" in nuclear_norm_source
    assert "singular_values.sum()" in nuclear_norm_source


def test_generate_torch_method_source_uses_semantic_bridges_for_selected_ops():
    module = _load_generator_module()

    cond_op = module._parse_func(
        "linalg_cond.out(Tensor self, Scalar? p=None, *, Tensor(a!) out) -> Tensor(a!)"
    )
    cond_source = module._generate_torch_method_source("linalg_cond", cond_op)
    assert "std::get<1>(at::linalg_svd(" in cond_source
    assert "singular_values.max() / singular_values.min()" in cond_source

    pool_op = module._parse_func(
        "mkldnn_adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)"
    )
    pool_source = module._generate_torch_method_source(
        "mkldnn_adaptive_avg_pool2d", pool_op
    )
    assert "at::adaptive_avg_pool2d(" in pool_source
    assert "mkldnn_adaptive_avg_pool2d_out" not in pool_source

    depthwise_op = module._parse_func(
        "_conv_depthwise2d.out(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, SymInt[2] padding, int[2] dilation, *, Tensor(a!) out) -> Tensor(a!)"
    )
    depthwise_source = module._generate_torch_method_source(
        "aten_conv_depthwise2d", depthwise_op
    )
    assert "at::conv2d(" in depthwise_source
    assert ".size(1))" in depthwise_source

    hspmm_op = module._parse_func(
        "hspmm.out(Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)"
    )
    hspmm_source = module._generate_torch_method_source("hspmm", hspmm_op)
    assert "at::hspmm(at_mat1.cpu(), at_mat2.cpu())" in hspmm_source
    assert ".to(at_mat1.device())" in hspmm_source

    sspaddmm_op = module._parse_func(
        "sspaddmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)"
    )
    sspaddmm_source = module._generate_torch_method_source("sspaddmm", sspaddmm_op)
    assert (
        "at::sspaddmm(at_input.cpu(), at_mat1.cpu(), at_mat2.cpu()," in sspaddmm_source
    )
    assert ".to(at_input.device())" in sspaddmm_source

    int_mm_op = module._parse_func(
        "_int_mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)"
    )
    int_mm_source = module._generate_torch_method_source("aten_int_mm", int_mm_op)
    assert (
        "at::matmul(at_input.cpu().to(at::kInt), at_mat2.cpu().to(at::kInt))"
        in int_mm_source
    )
    assert ".to(at_input.device())" in int_mm_source

    scaled_mm_op = module._parse_func(
        "_scaled_mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out, Tensor(b!) out_amax) -> (Tensor(a!), Tensor(b!))"
    )
    scaled_mm_source = module._generate_torch_method_source(
        "aten_scaled_mm", scaled_mm_op
    )
    assert "auto result = at::matmul(" in scaled_mm_source
    assert "result.abs().amax()" in scaled_mm_source

    scaled_mm_single_out_op = module._parse_func(
        "_scaled_mm.out(Tensor self, Tensor mat2, Tensor scale_a, Tensor scale_b, Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, bool use_fast_accum=False, *, Tensor(a!) out) -> Tensor(a!)"
    )
    scaled_mm_single_out_source = module._generate_torch_method_source(
        "aten_scaled_mm", scaled_mm_single_out_op
    )
    assert "auto result = at::matmul(" in scaled_mm_single_out_source
    assert "result.abs().amax()" not in scaled_mm_single_out_source


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
