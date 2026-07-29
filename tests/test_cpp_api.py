import os
import subprocess
import textwrap
from pathlib import Path

import pytest


def test_cpp_operator_call_instantiation_smoke(tmp_path):
    install_prefix = _install_prefix()
    include_dir = install_prefix / "include"
    library_dir = _library_dir(install_prefix)
    source = tmp_path / "add_smoke.cc"
    binary = tmp_path / "add_smoke"
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
    _run([str(binary)])


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


def test_cpp_operator_cache_fast_path(tmp_path):
    install_prefix = _install_prefix()
    include_dir = install_prefix / "include"
    library_dir = _library_dir(install_prefix)
    source_include_dir = Path(__file__).resolve().parents[1] / "src"
    source = tmp_path / "operator_cache_fast_path.cc"
    binary = tmp_path / "operator_cache_fast_path"
    source.write_text(_OPERATOR_CACHE_FAST_PATH_SOURCE)

    _run(
        [
            _compiler("CXX", "c++"),
            "-std=c++17",
            "-Werror",
            f"-I{source_include_dir}",
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


def _run(command):
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as error:
        pytest.skip(f"`{command[0]}` is not available: {error}")
    except subprocess.CalledProcessError as error:
        output = "\n".join((error.stdout, error.stderr)).strip()
        raise AssertionError(output) from error


_ADD_SMOKE_SOURCE = textwrap.dedent(
    r"""
    #include <infini/ops.h>

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


_OPERATOR_CACHE_FAST_PATH_SOURCE = textwrap.dedent(
    r"""
    #include <operator.h>

    #include <cstddef>
    #include <stdexcept>
    #include <vector>

    namespace infini::ops {

    inline std::size_t key_build_count{0};
    inline std::size_t construction_count{0};
    inline std::size_t invocation_count{0};
    inline bool fail_construction{false};
    inline std::size_t legacy_key_build_count{0};
    inline std::size_t legacy_construction_count{0};
    inline std::size_t legacy_invocation_count{0};

    class CacheProbe : public Operator<CacheProbe> {
     public:
      virtual void operator()(const Tensor tensor) const = 0;
    };

    class LegacyCacheProbe : public Operator<LegacyCacheProbe> {
     public:
      virtual void operator()(const Tensor tensor) const = 0;
    };

    template <>
    struct CacheKeyBuilder<CacheProbe> {
      detail::CacheKey operator()(const Config& config,
                                  const Tensor tensor) const {
        ++key_build_count;
        return detail::CacheKey::Build(config.implementation_index(), tensor);
      }

      bool Matches(const detail::CacheKey& key, const Config& config,
                   const Tensor tensor) const {
        return key.Matches(config.implementation_index(), tensor);
      }
    };

    template <>
    struct CacheKeyBuilder<LegacyCacheProbe> {
      detail::CacheKey operator()(const Config& config,
                                  const Tensor tensor) const {
        ++legacy_key_build_count;
        return detail::CacheKey::Build(config.implementation_index(), tensor);
      }
    };

    template <>
    class Operator<CacheProbe, Device::Type::kCpu, 0> : public CacheProbe {
     public:
      explicit Operator(const Tensor /*tensor*/) {
        if (fail_construction) {
          throw std::runtime_error("requested constructor failure");
        }
        ++construction_count;
      }

      void operator()(const Tensor /*tensor*/) const override {
        ++invocation_count;
      }
    };

    template <>
    class Operator<LegacyCacheProbe, Device::Type::kCpu, 0>
        : public LegacyCacheProbe {
     public:
      explicit Operator(const Tensor /*tensor*/) {
        ++legacy_construction_count;
      }

      void operator()(const Tensor /*tensor*/) const override {
        ++legacy_invocation_count;
      }
    };

    }  // namespace infini::ops

    int main() {
      using infini::ops::CacheProbe;
      using infini::ops::Config;
      using infini::ops::DataType;
      using infini::ops::Device;
      using infini::ops::Handle;
      using infini::ops::LegacyCacheProbe;
      using infini::ops::Tensor;

      float data[11]{};
      const Device device{Device::Type::kCpu};
      const DataType dtype{DataType::kFloat32};
      const Tensor first{data, Tensor::Shape{2}, dtype, device};
      const Tensor same_metadata{data + 2, Tensor::Shape{2}, dtype, device};
      const Tensor different_metadata{data + 4, Tensor::Shape{3}, dtype,
                                      device};
      const Tensor failing_metadata{data + 7, Tensor::Shape{4}, dtype,
                                    device};
      const Handle handle;
      const Config config;

      CacheProbe::Call(handle, config, first);
      CacheProbe::Call(handle, config, same_metadata);
      CacheProbe::Call(handle, config, different_metadata);
      CacheProbe::Call(handle, config, same_metadata);
      CacheProbe::Call(handle, config, first);

      if (infini::ops::key_build_count != 2 ||
          infini::ops::construction_count != 2 ||
          infini::ops::invocation_count != 5) {
        return 1;
      }

      CacheProbe::clear_cache();
      CacheProbe::Call(handle, config, first);
      if (infini::ops::key_build_count != 3 ||
          infini::ops::construction_count != 3 ||
          infini::ops::invocation_count != 6) {
        return 2;
      }

      infini::ops::fail_construction = true;
      try {
        CacheProbe::Call(handle, config, failing_metadata);
        return 3;
      } catch (const std::runtime_error&) {
      }

      infini::ops::fail_construction = false;
      CacheProbe::Call(handle, config, failing_metadata);
      if (infini::ops::key_build_count != 5 ||
          infini::ops::construction_count != 4 ||
          infini::ops::invocation_count != 7) {
        return 4;
      }

      const std::vector<Tensor> first_group{first};
      const std::vector<Tensor> second_group{different_metadata};
      const auto grouped_key = infini::ops::detail::CacheKey::Build(
          first_group, second_group);
      if (!grouped_key.Matches(first_group, second_group)) {
        return 5;
      }

      const std::vector<Tensor> combined_group{first, different_metadata};
      const std::vector<Tensor> empty_group;
      if (grouped_key.Matches(combined_group, empty_group)) {
        return 6;
      }

      LegacyCacheProbe::Call(handle, config, first);
      LegacyCacheProbe::Call(handle, config, same_metadata);
      if (infini::ops::legacy_key_build_count != 2 ||
          infini::ops::legacy_construction_count != 1 ||
          infini::ops::legacy_invocation_count != 2) {
        return 7;
      }

      return 0;
    }
    """
).lstrip()
