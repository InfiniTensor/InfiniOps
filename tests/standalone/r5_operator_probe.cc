#include <infini/ops.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using infini::ops::Tensor;

Tensor view(const py::object& tensor) {
  Tensor::Shape shape;
  Tensor::Strides strides;
  for (const auto& value : tensor.attr("shape")) {
    shape.push_back(py::cast<size_t>(value));
  }
  for (const auto& value : tensor.attr("stride")()) {
    strides.push_back(py::cast<ptrdiff_t>(value));
  }
  const auto dtype_name = py::str(tensor.attr("dtype")).cast<std::string>();
  const auto dtype =
      dtype_name == "torch.int64"      ? infini::ops::DataType::kInt64
      : dtype_name == "torch.bfloat16" ? infini::ops::DataType::kBFloat16
      : dtype_name == "torch.float16"  ? infini::ops::DataType::kFloat16
                                       : infini::ops::DataType::kFloat32;
  return Tensor(
      reinterpret_cast<void*>(tensor.attr("data_ptr")().cast<uintptr_t>()),
      shape, dtype,
      infini::ops::Device{infini::ops::Device::Type::kIluvatar, 0}, strides);
}

PYBIND11_MODULE(r5_operator_probe, module) {
  module.def("rope", [](py::object query, py::object key, py::object positions,
                        py::object cache, bool neox, bool inverse,
                        uintptr_t stream) {
    infini::ops::Handle handle;
    handle.set_stream(reinterpret_cast<void*>(stream));
    std::optional<Tensor> k;
    if (!key.is_none()) k.emplace(view(key));
    infini::ops::RotaryEmbedding::Call(
        handle, infini::ops::Config::Automatic(), view(positions), view(query),
        k, view(cache), int64_t{128}, neox, int64_t{0}, inverse);
  });
  module.def("run", [](const std::string& operation, py::object input,
                       py::object other, py::object output, uintptr_t stream) {
    infini::ops::Handle handle;
    handle.set_stream(reinterpret_cast<void*>(stream));
    auto x = view(input);
    auto y = view(output);
    const auto config = infini::ops::Config::Automatic();
    if (operation == "softmax") {
      infini::ops::Softmax::Call(handle, config, x, int64_t{-1},
                                 std::optional<infini::ops::DataType>{}, y);
    } else if (operation == "swiglu") {
      infini::ops::SiluAndMul::Call(handle, config, x, y);
    } else if (operation == "add") {
      auto z = view(other);
      infini::ops::Add::Call(handle, config, x, z, y);
    } else {
      throw std::invalid_argument("Unknown probe operation.");
    }
  });
}
