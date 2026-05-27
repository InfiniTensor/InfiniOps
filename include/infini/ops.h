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

typedef struct InfiniOpsStreamPrivate* InfiniOpsStream;
typedef struct InfiniOpsTensorPrivate* InfiniOpsTensor;
typedef struct InfiniOpsHandlePrivate* InfiniOpsHandle;
typedef struct InfiniOpsConfigPrivate* InfiniOpsConfig;

INFINI_OPS_API InfiniOpsStatus infiniOpsGetLastError(char* buffer,
                                                     size_t capacity,
                                                     size_t* required_size);

INFINI_OPS_API InfiniOpsStatus infiniOpsCreateTensor(InfiniOpsTensor* tensor);

INFINI_OPS_API InfiniOpsStatus infiniOpsDestroyTensor(InfiniOpsTensor tensor);

INFINI_OPS_API InfiniOpsStatus infiniOpsSetTensorData(InfiniOpsTensor tensor,
                                                      void* data);

INFINI_OPS_API InfiniOpsStatus infiniOpsGetTensorData(InfiniOpsTensor tensor,
                                                      void** data);

INFINI_OPS_API InfiniOpsStatus
infiniOpsSetTensorByteSize(InfiniOpsTensor tensor, size_t byte_size);

INFINI_OPS_API InfiniOpsStatus
infiniOpsGetTensorByteSize(InfiniOpsTensor tensor, size_t* byte_size);

INFINI_OPS_API InfiniOpsStatus
infiniOpsSetTensorDataType(InfiniOpsTensor tensor, InfiniOpsDataType data_type);

INFINI_OPS_API InfiniOpsStatus infiniOpsGetTensorDataType(
    InfiniOpsTensor tensor, InfiniOpsDataType* data_type);

INFINI_OPS_API InfiniOpsStatus infiniOpsSetTensorDeviceType(
    InfiniOpsTensor tensor, InfiniOpsDeviceType device_type);

INFINI_OPS_API InfiniOpsStatus infiniOpsGetTensorDeviceType(
    InfiniOpsTensor tensor, InfiniOpsDeviceType* device_type);

INFINI_OPS_API InfiniOpsStatus infiniOpsSetTensorShape(InfiniOpsTensor tensor,
                                                       int32_t rank,
                                                       const int64_t* shape);

INFINI_OPS_API InfiniOpsStatus infiniOpsGetTensorShape(InfiniOpsTensor tensor,
                                                       int32_t* rank,
                                                       const int64_t** shape);

INFINI_OPS_API InfiniOpsStatus infiniOpsSetTensorStride(InfiniOpsTensor tensor,
                                                        const int64_t* stride);

INFINI_OPS_API InfiniOpsStatus
infiniOpsClearTensorStride(InfiniOpsTensor tensor);

INFINI_OPS_API InfiniOpsStatus infiniOpsGetTensorStride(InfiniOpsTensor tensor,
                                                        const int64_t** stride);

INFINI_OPS_API InfiniOpsStatus infiniOpsCreateHandle(InfiniOpsHandle* handle);

INFINI_OPS_API InfiniOpsStatus infiniOpsDestroyHandle(InfiniOpsHandle handle);

INFINI_OPS_API InfiniOpsStatus infiniOpsSetHandleStream(InfiniOpsHandle handle,
                                                        InfiniOpsStream stream);

INFINI_OPS_API InfiniOpsStatus
infiniOpsGetHandleStream(InfiniOpsHandle handle, InfiniOpsStream* stream);

INFINI_OPS_API InfiniOpsStatus infiniOpsSetHandleWorkspace(
    InfiniOpsHandle handle, void* workspace, size_t byte_size);

INFINI_OPS_API InfiniOpsStatus infiniOpsGetHandleWorkspace(
    InfiniOpsHandle handle, void** workspace, size_t* byte_size);

INFINI_OPS_API InfiniOpsStatus infiniOpsCreateConfig(InfiniOpsConfig* config);

INFINI_OPS_API InfiniOpsStatus infiniOpsDestroyConfig(InfiniOpsConfig config);

INFINI_OPS_API InfiniOpsStatus infiniOpsSetConfigImplementationIndex(
    InfiniOpsConfig config, size_t implementation_index);

INFINI_OPS_API InfiniOpsStatus infiniOpsGetConfigImplementationIndex(
    InfiniOpsConfig config, size_t* implementation_index);

#include <infini/c_ops.h>

#ifdef __cplusplus
}
#endif

#endif  // INFINI_OPS_H_
