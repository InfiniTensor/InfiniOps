#ifndef INFINI_OPS_CAMBRICON_COMMON_H_
#define INFINI_OPS_CAMBRICON_COMMON_H_

#include <cnnl.h>
#include <cnrt.h>

#include "data_type.h"
#include "device.h"

#define NRAM_MAX_SIZE (1024 * 240)

#ifdef __BANG__

namespace infini::ops::reduce {

constexpr int batch_size = 128 / sizeof(float);

__mlu_func__ void SumInternal(float* dst, float* src, int max_batch) {
  const int width = max_batch / batch_size;

  if (width >= 4) {
    __bang_sumpool(dst, src, batch_size, 1, width, 1, width, 1, 1);
    __bang_reduce_sum(dst, dst, batch_size);
  } else {
    float sum = 0.0f;
    for (int i = 0; i < max_batch; ++i) {
      sum += src[i];
    }
    dst[0] = sum;
  }
}

template <typename T>
__mlu_func__ void SumTyped(float *result, T *data, size_t len) {
  if constexpr (std::is_same_v<T, __half>) {
    __bang_half2float((float *)data, reinterpret_cast<half *>(data) + len, len);
    SumInternal(result, (float *)data, len);
  } else if constexpr (std::is_same_v<T, __bang_bfloat16>) {
    __bang_bfloat162float((float *)data, data + len, len);
    SumInternal(result, (float *)data, len);
  } else {
    SumInternal(result, data, len);
  }
}

template <typename T>
__mlu_func__ float Sum(const T *source, T *src, float *dst, int num_elements,
                       int max_batch) {
  float res = 0.0f;
  int offset = (sizeof(T) == 2 ? max_batch : 0);

  size_t processed = 0;
  while (processed < num_elements) {
    size_t curr_batch = std::min<size_t>(max_batch, num_elements - processed);

    if (curr_batch < max_batch) {
      __bang_write_value(src, max_batch + offset, 0);
    }

    __memcpy(src + offset, source + processed, curr_batch * sizeof(T),
             GDRAM2NRAM);
    SumTyped(dst, src, max_batch);
    res += dst[0];
    processed += curr_batch;
  }

  return res;
}

template <typename T>
__mlu_func__ float SumBatched(const T *source, T *src, float *dst,
                              int num_elements, int max_batch) {
  constexpr int min_vector_size = 32;

  if (num_elements < min_vector_size) {
    return Sum(source, src, dst, num_elements, max_batch);
  }

  float res = 0.0f;
  int offset = (sizeof(T) == 2 ? max_batch : 0);

  size_t processed = 0;
  while (processed < num_elements) {
    size_t curr_batch = std::min<size_t>(max_batch, num_elements - processed);
    size_t aligned_batch = (curr_batch / batch_size) * batch_size;
    size_t remainder = curr_batch % batch_size;

    // Ensure NRAM buffer is zeroed.
    __bang_write_value(src, max_batch + offset, 0);

    // Copy data to NRAM.
    __memcpy(src + offset, source + processed, curr_batch * sizeof(T),
             GDRAM2NRAM);

    if constexpr (std::is_same_v<T, __half>) {
      __bang_half2float((float *)(src + offset),
                        reinterpret_cast<half *>(src) + offset, curr_batch);
    } else if constexpr (std::is_same_v<T, __bang_bfloat16>) {
      __bang_bfloat162float((float *)(src + offset), src + offset, curr_batch);
    }

    if (aligned_batch > 0) {
      SumInternal(dst, (float *)(src + offset), aligned_batch);
      res += dst[0];
    }
    if (remainder > 0) {
      for (size_t i = aligned_batch; i < curr_batch; ++i) {
        res += ((float *)(src + offset))[i];
      }
    }

    processed += curr_batch;
  }

  return res;
}

__mlu_func__ void MaxInternal(float *dst, float *src, int max_batch) {
  __bang_maxpool(dst, src, batch_size, 1, max_batch / batch_size, 1,
                 max_batch / batch_size, 1, 1);
  __bang_argmax(dst, dst, batch_size);
}

template <typename T>
__mlu_func__ void MaxTyped(float *result, T *data, size_t len) {
  if constexpr (std::is_same_v<T, __half>) {
    __bang_half2float((float *)data, reinterpret_cast<half *>(data) + len, len);
    MaxInternal(result, (float *)data, len);
  } else if constexpr (std::is_same_v<T, __bang_bfloat16>) {
    __bang_bfloat162float((float *)data, data + len, len);
    MaxInternal(result, (float *)data, len);
  } else {
    MaxInternal(result, data, len);
  }
}

template <typename T>
__mlu_func__ float Max(const T *source, T *src, float *dst, int num_elements,
                       int max_batch) {
  float max_val = -INFINITY;
  int offset = (sizeof(T) == 2 ? max_batch : 0);

  size_t processed = 0;
  while (processed < num_elements) {
    size_t curr_batch = std::min<size_t>(max_batch, num_elements - processed);

    if (curr_batch < max_batch) {
      __bang_write_value(src, max_batch + offset, 0);
    }

    __memcpy(src + offset, source + processed, curr_batch * sizeof(T),
             GDRAM2NRAM);
    MaxTyped(dst, src, max_batch);
    max_val = std::max(max_val, dst[0]);
    processed += curr_batch;
  }

  return max_val;
}

template <typename T>
__mlu_func__ float MaxBatched(const T *source, T *src, float *dst,
                              int num_elements, int max_batch) {
  constexpr int min_vector_size = 32;

  if (num_elements < min_vector_size) {
    return Max(source, src, dst, num_elements, max_batch);
  }

  float max_val = -INFINITY;
  int offset = (sizeof(T) == 2 ? max_batch : 0);

  size_t processed = 0;
  while (processed < num_elements) {
    size_t curr_batch = std::min<size_t>(max_batch, num_elements - processed);

    if (curr_batch < max_batch) {
      __bang_write_value(src, max_batch + offset, 0);
    }

    __memcpy(src + offset, source + processed, curr_batch * sizeof(T),
             GDRAM2NRAM);
    MaxTyped(dst, src, max_batch);
    max_val = std::max(max_val, dst[0]);
    processed += curr_batch;
  }

  return max_val;
}

}  // namespace infini::ops::reduce

#endif  // __BANG__

namespace infini::ops::cnnl_utils {

inline cnnlDataType_t GetDataType(DataType dtype) {
  switch (dtype) {
    case DataType::kInt8:
      return CNNL_DTYPE_INT8;
    case DataType::kUInt8:
      return CNNL_DTYPE_UINT8;
    case DataType::kInt32:
      return CNNL_DTYPE_INT32;
    case DataType::kInt64:
      return CNNL_DTYPE_INT64;
    case DataType::kFloat16:
      return CNNL_DTYPE_HALF;
    case DataType::kFloat32:
      return CNNL_DTYPE_FLOAT;
    case DataType::kBFloat16:
      return CNNL_DTYPE_BFLOAT16;
    case DataType::kFloat64:
      return CNNL_DTYPE_DOUBLE;
    default:
      return CNNL_DTYPE_INVALID;
  }
}

}  // namespace infini::ops::cnnl_utils

namespace infini::ops::cnrt_utils {

inline void GetLaunchConfig(const Device& device, int* core_per_cluster,
                            int* cluster_count) {
  int device_id = device.index();
  cnrtDeviceGetAttribute(cluster_count, cnrtAttrClusterCount, device_id);
  cnrtDeviceGetAttribute(core_per_cluster, cnrtAttrMcorePerCluster, device_id);
}

}  // namespace infini::ops::cnrt_utils

#endif
