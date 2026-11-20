// Test adapter for workspace-taking Clover operators with explicit
// implementation index selection, mirroring clover_workspace_smoke.cpp.
#include <infini/ops.h>
#include <pybind11/pybind11.h>

#include "pybind11_utils.h"

namespace py = pybind11;
using namespace infini::ops;

PYBIND11_MODULE(clover_native_adapter, m) {
  m.def("coreset", [](py::object logits, size_t width, double top_p,
                      py::object out, py::object workspace, uintptr_t stream,
                      size_t implementation_index) {
    Handle handle;
    handle.set_stream(reinterpret_cast<void*>(stream));
    auto w = TensorFromPybind11Handle(workspace);
    handle.set_workspace(w.data());
    handle.set_workspace_size_in_bytes(w.numel() * w.element_size());
    Config config;
    config.set_implementation_index(implementation_index);
    CloverCoreset::Call(handle, config, TensorFromPybind11Handle(logits),
                        width, top_p, TensorFromPybind11Handle(out));
  });
}
