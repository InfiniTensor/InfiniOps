#ifndef INFINI_OPS_TORCH_PYBIND11__H_
#define INFINI_OPS_TORCH_PYBIND11__H_

#include <optional>
#include <string>

#include "data_type.h"
#include "tensor.h"

namespace infini::ops {

// Torch-independent metadata returned by the host-compiled source adapter.
struct AtenTensorMetadata {
  void* data;

  Tensor::Shape shape;

  Tensor::Strides strides;

  DataType dtype;

  std::string device_type;

  int device_index;
};

std::optional<AtenTensorMetadata> TryAtenTensorMetadataFromPyObject(
    void* py_object);

}  // namespace infini::ops

#endif
