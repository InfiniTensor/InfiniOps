#include <infini/ops.h>
#include <pybind11/pybind11.h>

#include "pybind11_utils.h"

namespace py = pybind11;
using namespace infini::ops;

PYBIND11_MODULE(workspace_ops, m) {
  m.def("moe", [](py::list tensors, size_t block_size, py::object workspace,
                  uintptr_t stream) {
    if (tensors.size() != 9)
      throw py::value_error("Expected nine MoE tensors.");
    Handle handle;
    handle.set_stream(reinterpret_cast<void*>(stream));
    auto scratch = TensorFromPybind11Handle(workspace);
    handle.set_workspace(scratch.data());
    handle.set_workspace_size_in_bytes(scratch.numel() *
                                       scratch.element_size());
    const auto tensor = [&](int i) {
      return TensorFromPybind11Handle(tensors[i]);
    };
    MoeFusedDense::Call(handle,
                        MoeFusedDense::DefaultConfig(Device::Type::kHygon),
                        tensor(0), tensor(1), tensor(2), tensor(3), tensor(4),
                        tensor(5), tensor(6), tensor(7), tensor(8), block_size,
                        std::optional<Tensor>{}, std::optional<Tensor>{});
  });
  m.def("coreset", [](py::object logits, size_t width, double top_p,
                      py::object out, py::object workspace, uintptr_t stream) {
    Handle handle;
    handle.set_stream(reinterpret_cast<void*>(stream));
    auto w = TensorFromPybind11Handle(workspace);
    handle.set_workspace(w.data());
    handle.set_workspace_size_in_bytes(w.numel());
    CloverCoreset::Call(handle,
                        CloverCoreset::DefaultConfig(Device::Type::kHygon),
                        TensorFromPybind11Handle(logits), width, top_p,
                        TensorFromPybind11Handle(out));
  });
}
