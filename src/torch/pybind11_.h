#ifndef INFINI_OPS_TORCH_PYBIND11__H_
#define INFINI_OPS_TORCH_PYBIND11__H_

#include <memory>
#include <string>

#include "data_type.h"
#include "tensor.h"

namespace infini::ops {

// Torch-independent metadata returned by the host-compiled bridge. Keeping
// ATen headers out of generated bindings lets vendor compilers build them.
struct AtenTensorMetadata {
  void* data;
  Tensor::Shape shape;
  Tensor::Strides strides;
  DataType dtype;
  std::string device_type;
  int device_index;
  std::shared_ptr<const void> source_handle;
};

AtenTensorMetadata AtenTensorMetadataFromPyObject(void* py_object);

}  // namespace infini::ops

#endif
