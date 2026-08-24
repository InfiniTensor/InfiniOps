import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest


def test_cpp_operator_call_instantiation_smoke(tmp_path):
    install_prefix = _install_prefix()
    source_dir = tmp_path / "source"
    build_dir = tmp_path / "build"
    source_dir.mkdir()
    source = source_dir / "add_smoke.cc"
    source.write_text(_ADD_SMOKE_SOURCE)
    (source_dir / "CMakeLists.txt").write_text(_CMAKE_PACKAGE_SMOKE_PROJECT)

    _run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={install_prefix}",
            f"-DCMAKE_CXX_COMPILER={_compiler('CXX', 'c++')}",
        ]
    )
    _run(["cmake", "--build", str(build_dir)])
    _run([str(build_dir / "add_smoke")])


def test_cpp_operator_call_trace_is_json(tmp_path):
    install_prefix = _install_prefix()
    include_dir = install_prefix / "include"
    library_dir = _library_dir(install_prefix)
    source = tmp_path / "add_trace.cc"
    binary = tmp_path / "add_trace"
    source.write_text(_ADD_SMOKE_SOURCE)

    _run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-Werror",
            f"-I{include_dir}",
            str(source),
            f"-L{library_dir}",
            "-linfiniops",
            "-linfinirt",
            f"-Wl,-rpath,{library_dir}",
            "-o",
            str(binary),
        ]
    )
    env = os.environ.copy()
    env["INFINI_OPS_TRACE_CALLS"] = "1"
    result = _run([str(binary)], env=env)

    prefix = "[INFINI_OPS_TRACE_CALLS] "
    trace_lines = [
        line for line in result.stderr.splitlines() if line.startswith(prefix)
    ]
    assert len(trace_lines) == 2
    assert [json.loads(line.removeprefix(prefix)) for line in trace_lines] == [
        {"operator_name": "Add", "device_type": "cpu", "implementation": 0},
        {"operator_name": "Add", "device_type": "cpu", "implementation": 0},
    ]


def test_cpp_returning_call_smoke(tmp_path):
    install_prefix = _install_prefix()
    include_dir = install_prefix / "include"
    library_dir = _library_dir(install_prefix)
    source = tmp_path / "add_return_smoke.cc"
    binary = tmp_path / "add_return_smoke"
    source.write_text(_ADD_RETURN_SMOKE_SOURCE)

    _run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-Werror",
            f"-I{include_dir}",
            str(source),
            f"-L{library_dir}",
            "-linfiniops",
            "-linfinirt",
            f"-Wl,-rpath,{library_dir}",
            "-o",
            str(binary),
        ]
    )
    _run([str(binary)])


def test_cpp_configless_calls_use_first_active_implementation(tmp_path):
    install_prefix = _install_prefix()
    include_dir = install_prefix / "include"
    library_dir = _library_dir(install_prefix)
    source = tmp_path / "configless_active_implementation.cc"
    binary = tmp_path / "configless_active_implementation"
    source.write_text(_CONFIGLESS_ACTIVE_IMPLEMENTATION_SOURCE)

    _run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-Werror",
            f"-I{include_dir}",
            str(source),
            f"-L{library_dir}",
            "-linfiniops",
            "-linfinirt",
            f"-Wl,-rpath,{library_dir}",
            "-o",
            str(binary),
        ]
    )
    _run([str(binary)])


def test_cpp_polymorphic_context_smoke(tmp_path):
    install_prefix = _install_prefix()
    include_dir = install_prefix / "include"
    library_dir = _library_dir(install_prefix)
    source = tmp_path / "polymorphic_context.cc"
    binary = tmp_path / "polymorphic_context"
    source.write_text(_POLYMORPHIC_CONTEXT_SOURCE)

    _run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-Werror",
            f"-I{include_dir}",
            str(source),
            f"-L{library_dir}",
            "-linfiniops",
            "-linfinirt",
            f"-Wl,-rpath,{library_dir}",
            "-o",
            str(binary),
        ]
    )
    _run([str(binary)])


@pytest.mark.parametrize(
    "header",
    (
        "base/clamp.h",
        "base/moe_sum.h",
        "base/scaled_dot_product_attention.h",
    ),
)
def test_cpp_base_headers_compile_with_metadata_views(tmp_path, header):
    install_prefix = _install_prefix()
    include_dir = install_prefix / "include"
    source = tmp_path / f"{Path(header).stem}_metadata_view.cc"
    source.write_text(f"#include <{header}>\n\nint main() {{ return 0; }}\n")

    _run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-Werror",
            "-UNDEBUG",
            "-fsyntax-only",
            f"-I{include_dir}",
            str(source),
        ]
    )


def _install_prefix():
    prefix = os.environ.get("INFINI_OPS_INSTALL_PREFIX")

    if prefix:
        return Path(prefix)

    pytest.skip("`INFINI_OPS_INSTALL_PREFIX` is not set.")


def _library_dir(prefix):
    for name in ("lib", "lib64"):
        library_dir = prefix / name
        if (library_dir / "libinfiniops.so").exists():
            return library_dir

    pytest.skip(f"`libinfiniops.so` was not found under `{prefix}`.")


def _compiler(env_name, default):
    compiler = os.environ.get(env_name, default)

    if not compiler:
        pytest.skip(f"`{env_name}` is not configured.")

    return compiler


def _run(command, **kwargs):
    try:
        return subprocess.run(
            command, check=True, text=True, capture_output=True, **kwargs
        )
    except FileNotFoundError as error:
        pytest.skip(f"`{command[0]}` is not available: {error}")
    except subprocess.CalledProcessError as error:
        output = "\n".join((error.stdout, error.stderr)).strip()
        raise AssertionError(output) from error


_CMAKE_PACKAGE_SMOKE_PROJECT = textwrap.dedent(
    """\
    cmake_minimum_required(VERSION 3.18)
    project(infiniops_cpp_smoke LANGUAGES CXX)

    find_package(InfiniOps CONFIG REQUIRED)

    add_executable(add_smoke add_smoke.cc)
    target_compile_features(add_smoke PRIVATE cxx_std_17)
    target_compile_options(add_smoke PRIVATE -Werror)
    target_link_libraries(add_smoke PRIVATE InfiniOps::infiniops)
    """
)


_ADD_SMOKE_SOURCE = textwrap.dedent(
    r"""
    #include <infini/ops.h>

    #include <algorithm>
    #include <cmath>

    int main() {
      float input_data[3] = {1.0f, 2.0f, 3.0f};
      float other_data[3] = {4.0f, 5.0f, 6.0f};
      float output_data[3] = {0.0f, 0.0f, 0.0f};

      const infini::ops::Tensor::Shape shape{3};
      const infini::ops::Device device{infini::ops::Device::Type::kCpu};
      const infini::ops::DataType data_type{infini::ops::DataType::kFloat32};

      infini::ops::Tensor input(input_data, shape, data_type, device);
      infini::ops::Tensor other(other_data, shape, data_type, device);
      infini::ops::Tensor output(output_data, shape, data_type, device);
      infini::ops::Handle handle;
      infini::ops::Config config;

      const auto active_implementations =
          infini::ops::Add::active_implementation_indices(
              infini::ops::Device::Type::kCpu);
      if (std::find(active_implementations.begin(), active_implementations.end(),
                    0) == active_implementations.end()) {
        return 1;
      }

      infini::ops::Add::Call(handle, config, input, other, output);

      if (std::fabs(output_data[0] - 5.0f) > 1e-6f ||
          std::fabs(output_data[1] - 7.0f) > 1e-6f ||
          std::fabs(output_data[2] - 9.0f) > 1e-6f) {
        return 1;
      }

      output_data[0] = 0.0f;
      output_data[1] = 0.0f;
      output_data[2] = 0.0f;

      infini::ops::Add::Call(input, other, output);

      if (std::fabs(output_data[0] - 5.0f) > 1e-6f ||
          std::fabs(output_data[1] - 7.0f) > 1e-6f ||
          std::fabs(output_data[2] - 9.0f) > 1e-6f) {
        return 1;
      }

      return 0;
    }
    """
).lstrip()


_ADD_RETURN_SMOKE_SOURCE = textwrap.dedent(
    r"""
    #include <infini/ops.h>

    #include <base/embedding.h>
    #include <base/scaled_dot_product_attention.h>
    #include <base/silu_and_mul.h>

    #include <cmath>
    #include <functional>
    #include <numeric>
    #include <stdexcept>
    #include <type_traits>
    #include <utility>
    #include <vector>

    class OwningTensor {
     public:
      using Shape = infini::ops::Tensor::Shape;
      using Strides = infini::ops::Tensor::Strides;

      OwningTensor(std::vector<float> data, Shape shape)
          : data_{std::move(data)},
            shape_{std::move(shape)},
            strides_{ContiguousStrides(shape_)},
            dtype_{infini::ops::DataType::kFloat32},
            device_{infini::ops::Device::Type::kCpu} {}

      static OwningTensor Empty(const Shape& shape, infini::ops::DataType dtype,
                                infini::ops::Device device) {
        if (dtype != infini::ops::DataType::kFloat32 ||
            device.type() != infini::ops::Device::Type::kCpu) {
          throw std::runtime_error("unexpected output metadata");
        }

        return OwningTensor(std::vector<float>(Numel(shape)), shape);
      }

      void* data() { return data_.data(); }

      const void* data() const { return data_.data(); }

      template <typename ShapeAccess = decltype(
                    std::declval<const infini::ops::Tensor&>().shape())>
      decltype(auto) shape() const {
        if constexpr (std::is_reference_v<ShapeAccess>) {
          return (shape_);
        } else {
          return ShapeAccess{shape_.data(), shape_.size()};
        }
      }

      template <typename StridesAccess = decltype(
                    std::declval<const infini::ops::Tensor&>().strides())>
      decltype(auto) strides() const {
        if constexpr (std::is_reference_v<StridesAccess>) {
          return (strides_);
        } else {
          return StridesAccess{strides_.data(), strides_.size()};
        }
      }

      Shape::value_type size(std::ptrdiff_t dim) const {
        const auto index = dim < 0
                               ? static_cast<std::ptrdiff_t>(shape_.size()) + dim
                               : dim;
        return shape_[static_cast<std::size_t>(index)];
      }

      infini::ops::DataType dtype() const { return dtype_; }

      infini::ops::Device device() const { return device_; }

     private:
      static std::size_t Numel(const Shape& shape) {
        return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                               std::multiplies<std::size_t>());
      }

      static Strides ContiguousStrides(const Shape& shape) {
        if (shape.empty()) {
          return {};
        }

        Strides strides(shape.size());
        strides.back() = 1;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(shape.size()) - 2;
             i >= 0; --i) {
          strides[static_cast<std::size_t>(i)] =
              strides[static_cast<std::size_t>(i + 1)] *
              static_cast<infini::ops::Tensor::Stride>(
                  shape[static_cast<std::size_t>(i + 1)]);
        }
        return strides;
      }

      std::vector<float> data_;
      Shape shape_;
      Strides strides_;
      infini::ops::DataType dtype_;
      infini::ops::Device device_;
    };

    int main() {
      OwningTensor input({1.0f, 2.0f, 3.0f}, {3});
      OwningTensor other({4.0f, 5.0f, 6.0f}, {3});

      auto output = infini::ops::Add::Call(input, other);
      const auto* data = static_cast<const float*>(output.data());

      if (output.shape() != OwningTensor::Shape{3}) {
        return 1;
      }
      if (std::fabs(data[0] - 5.0f) > 1e-6f ||
          std::fabs(data[1] - 7.0f) > 1e-6f ||
          std::fabs(data[2] - 9.0f) > 1e-6f) {
        return 1;
      }

      OwningTensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
      OwningTensor b({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2});

      auto c = infini::ops::Gemm::Call(a, b);
      const auto* c_data = static_cast<const float*>(c.data());

      if (c.shape() != OwningTensor::Shape{2, 2}) {
        return 1;
      }
      if (std::fabs(c_data[0] - 58.0f) > 1e-6f ||
          std::fabs(c_data[1] - 64.0f) > 1e-6f ||
          std::fabs(c_data[2] - 139.0f) > 1e-6f ||
          std::fabs(c_data[3] - 154.0f) > 1e-6f) {
        return 1;
      }

      OwningTensor silu_input(std::vector<float>(8), {2, 4});
      auto silu_output = infini::ops::SiluAndMul::MakeReturnValue(silu_input);
      if (silu_output.shape() != OwningTensor::Shape{2, 2}) {
        return 1;
      }

      OwningTensor embedding_input(std::vector<float>(6), {2, 3});
      OwningTensor embedding_weight(std::vector<float>(40), {10, 4});
      auto embedding_output = infini::ops::Embedding::MakeReturnValue(
          embedding_input, embedding_weight);
      if (embedding_output.shape() != OwningTensor::Shape{2, 3, 4}) {
        return 1;
      }

      OwningTensor query(std::vector<float>(24), {1, 2, 3, 4});
      OwningTensor key(std::vector<float>(40), {1, 2, 5, 4});
      OwningTensor value(std::vector<float>(60), {1, 2, 5, 6});
      auto attention_output =
          infini::ops::ScaledDotProductAttention::MakeReturnValue(query, key,
                                                                  value);
      if (attention_output.shape() != OwningTensor::Shape{1, 2, 3, 6}) {
        return 1;
      }

      return 0;
    }
    """
).lstrip()


_CONFIGLESS_ACTIVE_IMPLEMENTATION_SOURCE = textwrap.dedent(
    r"""
    #include <infini/ops.h>

    #include <cmath>
    #include <cstdint>
    #include <vector>

    namespace infini::ops {

    class ConfiglessSelection : public Operator<ConfiglessSelection> {
     public:
      ConfiglessSelection(const Tensor input, Tensor out) {}

      ConfiglessSelection(const std::vector<Tensor> inputs, Tensor out) {}

      virtual void operator()(const Tensor input, Tensor out) const = 0;

      virtual void operator()(const std::vector<Tensor> inputs,
                              Tensor out) const = 0;

      template <typename TensorLike>
      static auto MakeReturnValue(const TensorLike& input) {
        return TensorLike::Empty(input.shape(), input.dtype(), input.device());
      }
    };

    template <>
    class Operator<ConfiglessSelection, Device::Type::kCpu, 16>
        : public ConfiglessSelection {
     public:
      using ConfiglessSelection::ConfiglessSelection;

      void operator()(const Tensor input, Tensor out) const override {
        const auto* input_data = static_cast<const float*>(input.data());
        auto* out_data = static_cast<float*>(out.data());
        out_data[0] = input_data[0] + 16.0f;
      }

      void operator()(const std::vector<Tensor> inputs,
                      Tensor out) const override {
        const auto* first_data = static_cast<const float*>(inputs[0].data());
        const auto* second_data = static_cast<const float*>(inputs[1].data());
        auto* out_data = static_cast<float*>(out.data());
        out_data[0] = first_data[0] + second_data[0] + 16.0f;
      }
    };

    }  // namespace infini::ops

    int main() {
      float input_data = 1.0f;
      float second_data = 2.0f;
      float out_data = 0.0f;
      const infini::ops::Tensor::Shape shape{1};
      const infini::ops::Device device{infini::ops::Device::Type::kCpu};
      const infini::ops::DataType dtype{infini::ops::DataType::kFloat32};
      infini::ops::Tensor input(&input_data, shape, dtype, device);
      infini::ops::Tensor second(&second_data, shape, dtype, device);
      infini::ops::Tensor out(&out_data, shape, dtype, device);

      auto tensor_op = infini::ops::ConfiglessSelection::Make(input, out);
      (*tensor_op)(input, out);
      if (std::fabs(out_data - 17.0f) > 1e-6f) {
        return 1;
      }

      out_data = 0.0f;
      std::vector<infini::ops::Tensor> inputs{input, second};
      auto vector_op = infini::ops::ConfiglessSelection::Make(inputs, out);
      (*vector_op)(inputs, out);
      if (std::fabs(out_data - 19.0f) > 1e-6f) {
        return 2;
      }

      float cat_out_data[2] = {};
      const infini::ops::Tensor::Shape cat_shape{2};
      infini::ops::Tensor cat_out(cat_out_data, cat_shape, dtype, device);
      auto cat_op =
          infini::ops::Cat::Make(inputs, std::int64_t{0}, cat_out);
      (*cat_op)(inputs, std::int64_t{0}, cat_out);
      if (std::fabs(cat_out_data[0] - 1.0f) > 1e-6f ||
          std::fabs(cat_out_data[1] - 2.0f) > 1e-6f) {
        return 3;
      }

      float abs_input_data = -4.0f;
      float abs_out_data = 0.0f;
      infini::ops::Tensor abs_input(&abs_input_data, shape, dtype, device);
      infini::ops::Tensor abs_out(&abs_out_data, shape, dtype, device);
      auto abs_op = infini::ops::Abs::Make(abs_input, abs_out);
      (*abs_op)(abs_input, abs_out);
      if (std::fabs(abs_out_data - 4.0f) > 1e-6f) {
        return 4;
      }

      auto abs_op_from_rvalue = infini::ops::Abs::Make(
          abs_input,
          infini::ops::Tensor(&abs_out_data, shape, dtype, device));
      (*abs_op_from_rvalue)(abs_input, abs_out);
      if (std::fabs(abs_out_data - 4.0f) > 1e-6f) {
        return 5;
      }

      abs_out_data = 0.0f;
      infini::ops::Abs::Call(abs_input, abs_out);
      if (std::fabs(abs_out_data - 4.0f) > 1e-6f) {
        return 6;
      }

      return 0;
    }
    """
).lstrip()


_POLYMORPHIC_CONTEXT_SOURCE = textwrap.dedent(
    r"""
    #include <operator.h>

    #include <cstddef>
    #include <type_traits>

    namespace infini::ops {

    class IntermediateConfig
        : public Cloneable<Config, IntermediateConfig> {
     public:
      virtual int value() const { return -1; }
    };

    class DerivedConfig
        : public Cloneable<IntermediateConfig, DerivedConfig> {
     public:
      explicit DerivedConfig(int value) : value_{value} {}

      int value() const override { return value_; }

     private:
      int value_;
    };

    class IntermediateHandle
        : public Cloneable<Handle, IntermediateHandle> {
     public:
      virtual int value() const { return -1; }
    };

    class DerivedHandle
        : public Cloneable<IntermediateHandle, DerivedHandle> {
     public:
      explicit DerivedHandle(int value) : value_{value} {}

      int value() const override { return value_; }

     private:
      int value_;
    };

    class PolymorphicOwner final : public OperatorBase {
     public:
      int config_value() const {
        return static_cast<const IntermediateConfig&>(*config_ptr_).value();
      }

      std::size_t implementation_index() const {
        return config_ptr_->implementation_index();
      }

      int handle_value() const {
        return static_cast<const IntermediateHandle&>(*handle_ptr_).value();
      }

      void* handle_stream() const { return handle_ptr_->stream(); }
    };

    }  // namespace infini::ops

    int main() {
      using namespace infini::ops;

      static_assert(std::has_virtual_destructor_v<Config>);
      static_assert(std::has_virtual_destructor_v<Handle>);
      static_assert(
          std::is_same_v<DerivedConfig::Pointer, std::unique_ptr<Config>>);
      static_assert(
          std::is_same_v<DerivedHandle::Pointer, std::unique_ptr<Handle>>);

      PolymorphicOwner owner;

      {
        DerivedConfig config{17};
        config.set_implementation_index(3);
        owner.set_config(config);
        config.set_implementation_index(9);
      }

      if (owner.config_value() != 17) return 1;
      if (owner.implementation_index() != 3) return 2;

      int stream;
      {
        DerivedHandle handle{23};
        handle.set_stream(&stream);
        owner.set_handle(handle);
        handle.set_stream(nullptr);
      }

      if (owner.handle_value() != 23) return 3;
      if (owner.handle_stream() != &stream) return 4;

      return 0;
    }
    """
).lstrip()
