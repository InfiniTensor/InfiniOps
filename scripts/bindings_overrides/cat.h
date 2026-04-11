#ifndef INFINI_OPS_BINDINGS_CAT_H_
#define INFINI_OPS_BINDINGS_CAT_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "base/cat.h"
#include "config.h"
#include "pybind11_utils.h"

namespace py = pybind11;

namespace infini::ops {

inline std::vector<Tensor> TensorListFromPybind11(py::list list) {
  std::vector<Tensor> result;
  result.reserve(py::len(list));

  for (auto& item : list) {
    result.push_back(TensorFromPybind11Handle(item));
  }

  return result;
}

void BindCat(py::module& m) {
  using Self = Cat;

  py::class_<Self>(m, "Cat")
      .def(py::init([](py::object first_input, py::list rest_inputs,
                       int64_t dim, py::object out) {
        return std::unique_ptr<Self>{static_cast<Self*>(
            Self::make(TensorFromPybind11Handle(first_input),
                       TensorListFromPybind11(rest_inputs), dim,
                       TensorFromPybind11Handle(out))
                .release())};
      }))
      .def("__call__",
           [](const Self& self, py::object first_input, py::list rest_inputs,
              int64_t dim, py::object out) {
             return static_cast<const Operator<Self>&>(self)(
                 TensorFromPybind11Handle(first_input),
                 TensorListFromPybind11(rest_inputs), dim,
                 TensorFromPybind11Handle(out));
           })
      .def_static("active_implementation_indices",
                   [](const std::string& device) {
                     return Self::active_implementation_indices(
                         DeviceTypeFromString(device));
                   });

  m.def(
      "cat",
      [](py::object first_input, py::list rest_inputs, int64_t dim,
         py::object out, std::size_t implementation_index) {
        Config config;
        config.set_implementation_index(implementation_index);
        return Self::call({}, config, TensorFromPybind11Handle(first_input),
                          TensorListFromPybind11(rest_inputs), dim,
                          TensorFromPybind11Handle(out));
      },
      py::arg("first_input"), py::arg("rest_inputs"), py::arg("dim"),
      py::arg("out"), py::kw_only(), py::arg("implementation_index") = 0);
}

}  // namespace infini::ops

#endif
