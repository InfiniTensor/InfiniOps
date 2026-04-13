#ifndef INFINI_OPS_ASCEND_COMMON_H_
#define INFINI_OPS_ASCEND_COMMON_H_

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "ascend/data_type_.h"
#include "tensor.h"

namespace infini::ops::ascend {

// Build an `aclTensor` descriptor from an InfiniOps `Tensor`.
//
// When `transpose_last2` is true the last two dimensions are swapped in the
// descriptor (shape and strides) without copying data.  This is used by `Gemm`
// and `MatMul` to express a transpose via the view.
inline aclTensor* buildAclTensor(const Tensor& t,
                                 bool transpose_last2 = false) {
  std::vector<int64_t> shape(t.shape().begin(), t.shape().end());
  std::vector<int64_t> strides(t.strides().begin(), t.strides().end());

  if (transpose_last2 && shape.size() >= 2) {
    auto n = shape.size();
    std::swap(shape[n - 2], shape[n - 1]);
    std::swap(strides[n - 2], strides[n - 1]);
  }

  // Compute the minimum physical storage needed for this strided view.
  // For contiguous tensors this equals `numel()`; for non-contiguous (gapped)
  // tensors it may be larger; for broadcast (stride-0) tensors it may be
  // smaller.  Passing the view shape as the storage shape causes
  // "ViewShape overlap" errors in ACLNN for non-contiguous inputs.
  int64_t storage_elems = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) {
      storage_elems = 0;
      break;
    }
    if (strides[i] > 0 && shape[i] > 1) {
      storage_elems += static_cast<int64_t>(shape[i] - 1) * strides[i];
    }
  }
  std::vector<int64_t> storage_shape = {storage_elems};

  return aclCreateTensor(
      shape.data(), static_cast<int64_t>(shape.size()), ToAclDtype(t.dtype()),
      strides.data(),
      /*storageOffset=*/0, ACL_FORMAT_ND, storage_shape.data(),
      static_cast<int64_t>(storage_shape.size()), const_cast<void*>(t.data()));
}

}  // namespace infini::ops::ascend

#endif
