#include <infini/ops.h>
#include <pybind11/pybind11.h>

#include "native/cuda/hygon/ops/moe_fused_dense/kernel.h"
#include "pybind11_utils.h"

namespace py = pybind11;
using namespace infini::ops;

namespace {

// The nine tensors of the fused MoE expert computation, in operator order.
std::vector<Tensor> MoeTensors(const py::list& tensors) {
  if (tensors.size() != 9)
    throw py::value_error("Expected nine MoE tensors.");
  std::vector<Tensor> views;
  for (auto& tensor : tensors)
    views.push_back(TensorFromPybind11Handle(tensor));
  return views;
}

Config MoeConfig(std::optional<size_t> implementation) {
  auto config = MoeFusedDense::DefaultConfig(Device::Type::kHygon);
  if (implementation) config.set_implementation_index(*implementation);
  return config;
}

}  // namespace

PYBIND11_MODULE(workspace_ops, m) {
  m.def("moe",
        [](py::list tensors, size_t block_size, py::object workspace,
           uintptr_t stream, std::optional<size_t> implementation) {
          const auto views = MoeTensors(tensors);
          Handle handle;
          handle.set_stream(reinterpret_cast<void*>(stream));
          auto scratch = TensorFromPybind11Handle(workspace);
          handle.set_workspace(scratch.data());
          handle.set_workspace_size_in_bytes(scratch.numel() *
                                             scratch.element_size());
          MoeFusedDense::Call(handle, MoeConfig(implementation), views[0],
                              views[1], views[2], views[3], views[4], views[5],
                              views[6], views[7], views[8], block_size,
                              std::optional<Tensor>{}, std::optional<Tensor>{});
        },
        py::arg("tensors"), py::arg("block_size"), py::arg("workspace"),
        py::arg("stream"), py::arg("implementation") = std::nullopt);
  m.def("moe_workspace_size",
        [](py::list tensors, size_t block_size) {
          const auto views = MoeTensors(tensors);
          const std::optional<Tensor> none;
          return MoeFusedDense::Make(MoeConfig(std::nullopt), views[0],
                                     views[1], views[2], views[3], views[4],
                                     views[5], views[6], views[7], views[8],
                                     block_size, none, none)
              ->workspace_size_in_bytes();
        },
        py::arg("tensors"), py::arg("block_size"));
  m.def("moe_has_native_kernels",
        [] { return moe_fused_dense_detail::HasNativeKernels(); });
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
