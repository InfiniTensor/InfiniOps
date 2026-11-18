#ifndef INFINI_OPS_BASE_VALIDATION_H_
#define INFINI_OPS_BASE_VALIDATION_H_

#include <cassert>
#include <cstdint>
#include <initializer_list>

#include "data_type.h"
#include "tensor.h"

namespace infini::ops::validation {

inline void AssertSameDevice(std::initializer_list<Tensor> tensors) {
  if (tensors.size() == 0) return;

  const auto device = tensors.begin()->device();
  for (const auto& tensor : tensors) {
    assert(tensor.device() == device &&
           "Tensors must share a device type and index.");
  }
}

inline bool IsFloatingPoint(DataType dtype) {
  return dtype == DataType::kFloat16 || dtype == DataType::kBFloat16 ||
         dtype == DataType::kFloat32 || dtype == DataType::kFloat64;
}

inline int64_t NormalizeDim(int64_t dim, Tensor::Size ndim) {
  const auto rank = static_cast<int64_t>(ndim);

  assert(rank > 0 && "Dimension requires a non-scalar tensor.");

  assert(dim >= -rank && dim < rank && "Dimension is out of range.");

  return dim < 0 ? dim + rank : dim;
}

inline void AssertBroadcastsTo(const Tensor input, const Tensor output) {
  assert(input.ndim() <= output.ndim() &&
         "Input rank must not exceed output rank.");
  const auto offset = output.ndim() - input.ndim();
  for (Tensor::Size i = 0; i < input.ndim(); ++i) {
    assert((input.size(i) == 1 || input.size(i) == output.size(i + offset)) &&
           "Input must broadcast to output with right-aligned dimensions.");
  }
}

}  // namespace infini::ops::validation

#endif  // INFINI_OPS_BASE_VALIDATION_H_
