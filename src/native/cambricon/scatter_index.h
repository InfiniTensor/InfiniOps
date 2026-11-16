#ifndef INFINI_OPS_CAMBRICON_SCATTER_INDEX_H_
#define INFINI_OPS_CAMBRICON_SCATTER_INDEX_H_

#include "native/cambricon/common.h"
#include "tensor.h"

namespace infini::ops {

void NormalizeScatterCoordinates(const void* indices, void* normalized,
                                 const int64_t* dims, size_t count, size_t width,
                                 DataType dtype, cnrtQueue_t queue);

void NormalizeAxisIndices(const void* indices, void* normalized,
                          uint32_t dim, size_t count, DataType dtype,
                          cnrtQueue_t queue);

}  // namespace infini::ops

#endif
