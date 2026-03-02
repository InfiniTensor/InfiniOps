#ifndef INFINI_OPS_CAMBRICON_COMMON_H_
#define INFINI_OPS_CAMBRICON_COMMON_H_

#include <cnnl.h>

#include "data_type.h"

namespace infini::ops::cnnl_utils {

inline cnnlDataType_t GetDtype(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat16:
      return CNNL_DTYPE_HALF;
    case DataType::kFloat32:
      return CNNL_DTYPE_FLOAT;
    case DataType::kInt32:
      return CNNL_DTYPE_INT32;
    default:
      return CNNL_DTYPE_INVALID;
  }
}

}  // namespace infini::ops::cnnl_utils

#endif
