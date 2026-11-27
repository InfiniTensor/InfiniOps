#ifndef INFINI_OPS_MARS_FLASH_ATTN_COMMON_H_
#define INFINI_OPS_MARS_FLASH_ATTN_COMMON_H_

#include <flash_attn/flash_attn.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include "data_type.h"
#include "tensor.h"

namespace infini::ops::mars_flash_attn {

inline hcflashattnDataType_t ToHcDataType(DataType dtype) {
  switch (dtype) {
    case DataType::kFloat16:
      return HCFLASHATTN_DATATYPE_FP16;
    case DataType::kBFloat16:
      return HCFLASHATTN_DATATYPE_BF16;
    case DataType::kInt32:
      return HCFLASHATTN_DATATYPE_INT32;
    case DataType::kInt64:
      return HCFLASHATTN_DATATYPE_INT64;
    case DataType::kFloat32:
      return HCFLASHATTN_DATATYPE_FP32;
    case DataType::kFloat64:
      return HCFLASHATTN_DATATYPE_FP64;
    default:
      throw std::invalid_argument("unsupported HPCC FlashAttention dtype");
  }
}

inline Tensor_t MakeHcTensor(const Tensor& tensor) {
  void* data = const_cast<void*>(tensor.data());
  const auto dtype = ToHcDataType(tensor.dtype());
  const auto size = [&tensor](std::size_t index) {
    return static_cast<int>(tensor.size(index));
  };
  const auto stride = [&tensor](std::size_t index) {
    return static_cast<int>(tensor.stride(index));
  };
  switch (tensor.ndim()) {
    case 1:
      return make_tensor1d(data, dtype, size(0), stride(0));
    case 2:
      return make_tensor2d(data, dtype, size(0), size(1), stride(0), stride(1));
    case 3:
      return make_tensor3d(data, dtype, size(0), size(1), size(2), stride(0),
                           stride(1), stride(2));
    case 4:
      return make_tensor4d(data, dtype, size(0), size(1), size(2), size(3),
                           stride(0), stride(1), stride(2), stride(3));
    default:
      throw std::invalid_argument("HPCC FlashAttention requires 1D-4D tensors");
  }
}

inline void ReleaseHcTensor(Tensor_t& tensor) {
  if (tensor) {
    release_tensor(tensor);
    tensor = nullptr;
  }
}

inline hcStream_t ToHcStream(void* stream) {
  return reinterpret_cast<hcStream_t>(stream);
}

inline void CheckHcStatus(hcflashattnStatus_t status, const char* call) {
  if (status != HCFLASHATTN_STATUS_SUCCESS)
    throw std::runtime_error(std::string{"HPCC FlashAttention failed: "} +
                             call);
}

inline float SoftmaxScaleOrDefault(const std::optional<double>& scale,
                                   std::size_t head_size) {
  return static_cast<float>(
      scale.value_or(1.0 / std::sqrt(static_cast<double>(head_size))));
}

}  // namespace infini::ops::mars_flash_attn

#endif  // INFINI_OPS_MARS_FLASH_ATTN_COMMON_H_
