#ifndef INFINI_OPS_LINKED_TORCH_MOORE_MATE_H_
#define INFINI_OPS_LINKED_TORCH_MOORE_MATE_H_

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#include <optional>
#include <string>

namespace infini::ops::linked::torch::moore::detail {

namespace py = pybind11;

inline py::module_ ImportMateMhaInterface() {
  auto mate = py::module_::import("mate");
  const auto version = py::str(mate.attr("__version__")).cast<std::string>();
  const auto separator = version.find('+');
  TORCH_CHECK(version.substr(0, separator) == "0.2.5",
              "Mate 0.2.5 is required by the Moore FlashAttention provider, "
              "but found ",
              version);
  return py::module_::import("mate.mha_interface");
}

inline py::object OptionalTensorToPyObject(
    const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() ? py::cast(*tensor) : py::none();
}

}  // namespace infini::ops::linked::torch::moore::detail

#endif  // INFINI_OPS_LINKED_TORCH_MOORE_MATE_H_
