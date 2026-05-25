#ifndef INFINI_OPS_H_
#define INFINI_OPS_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <infini/functional_ops.h>

extern "C" {
#endif

#if defined(_WIN32)
#if defined(INFINI_OPS_BUILD_SHARED)
#define INFINI_OPS_API __declspec(dllexport)
#elif defined(INFINI_OPS_USE_SHARED)
#define INFINI_OPS_API __declspec(dllimport)
#else
#define INFINI_OPS_API
#endif
#else
#if defined(INFINI_OPS_BUILD_SHARED)
#define INFINI_OPS_API __attribute__((visibility("default")))
#else
#define INFINI_OPS_API
#endif
#endif

typedef enum InfiniOpsStatus {
  INFINI_OPS_STATUS_SUCCESS,
  INFINI_OPS_STATUS_INVALID_ARGUMENT,
  INFINI_OPS_STATUS_NOT_SUPPORTED,
  INFINI_OPS_STATUS_OUT_OF_MEMORY,
  INFINI_OPS_STATUS_INTERNAL_ERROR,
} InfiniOpsStatus;

typedef enum InfiniOpsDataType {
  INFINI_OPS_DATA_TYPE_INVALID,
  INFINI_OPS_DATA_TYPE_INT8,
  INFINI_OPS_DATA_TYPE_INT16,
  INFINI_OPS_DATA_TYPE_INT32,
  INFINI_OPS_DATA_TYPE_INT64,
  INFINI_OPS_DATA_TYPE_UINT8,
  INFINI_OPS_DATA_TYPE_UINT16,
  INFINI_OPS_DATA_TYPE_UINT32,
  INFINI_OPS_DATA_TYPE_UINT64,
  INFINI_OPS_DATA_TYPE_FLOAT16,
  INFINI_OPS_DATA_TYPE_BFLOAT16,
  INFINI_OPS_DATA_TYPE_FLOAT32,
  INFINI_OPS_DATA_TYPE_FLOAT64,
} InfiniOpsDataType;

typedef enum InfiniOpsDeviceType {
  INFINI_OPS_DEVICE_TYPE_INVALID,
  INFINI_OPS_DEVICE_TYPE_CPU,
  INFINI_OPS_DEVICE_TYPE_NVIDIA,
  INFINI_OPS_DEVICE_TYPE_CAMBRICON,
  INFINI_OPS_DEVICE_TYPE_ASCEND,
  INFINI_OPS_DEVICE_TYPE_METAX,
  INFINI_OPS_DEVICE_TYPE_MOORE,
  INFINI_OPS_DEVICE_TYPE_ILUVATAR,
} InfiniOpsDeviceType;

typedef struct InfiniOpsTensor {
  size_t structure_size;
  void* data;
  size_t byte_size;
  InfiniOpsDataType data_type;
  InfiniOpsDeviceType device_type;
  int32_t rank;
  const int64_t* shape;
  const int64_t* stride;
  uint64_t reserved[8];
} InfiniOpsTensor;

typedef struct InfiniOpsStreamPrivate* InfiniOpsStream;
typedef struct InfiniOpsHandlePrivate* InfiniOpsHandle;
typedef struct InfiniOpsConfigPrivate* InfiniOpsConfig;

typedef struct InfiniOpsHandleAttributes {
  size_t structure_size;
  InfiniOpsStream stream;
  void* workspace;
  size_t workspace_byte_size;
  uint64_t reserved[8];
} InfiniOpsHandleAttributes;

typedef struct InfiniOpsConfigAttributes {
  size_t structure_size;
  size_t implementation_index;
  uint64_t reserved[8];
} InfiniOpsConfigAttributes;

INFINI_OPS_API InfiniOpsStatus infiniOpsGetLastError(char* buffer,
                                                     size_t capacity,
                                                     size_t* required_size);

INFINI_OPS_API InfiniOpsStatus infiniOpsCreateHandle(
    const InfiniOpsHandleAttributes* attributes, InfiniOpsHandle* handle);

INFINI_OPS_API InfiniOpsStatus infiniOpsDestroyHandle(InfiniOpsHandle handle);

INFINI_OPS_API InfiniOpsStatus infiniOpsCreateConfig(
    const InfiniOpsConfigAttributes* attributes, InfiniOpsConfig* config);

INFINI_OPS_API InfiniOpsStatus infiniOpsDestroyConfig(InfiniOpsConfig config);

#include <infini/c_ops.h>

#ifdef __cplusplus
}
#endif

#endif  // INFINI_OPS_H_
