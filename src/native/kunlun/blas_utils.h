#ifndef INFINI_OPS_KUNLUN_BLAS_UTILS_H_
#define INFINI_OPS_KUNLUN_BLAS_UTILS_H_

#include "data_type.h"
#include "native/cuda/blas_utils.h"
#include "native/kunlun/blas.h"

namespace infini::ops {

template <>
struct BlasUtils<Device::Type::kKunlun> {
  static auto GetDataType(DataType dtype) {
    switch (dtype) {
      case DataType::kFloat16:
        return Blas<Device::Type::kKunlun>::R_16F;
      case DataType::kBFloat16:
        return Blas<Device::Type::kKunlun>::R_16BF;
      case DataType::kFloat32:
        return Blas<Device::Type::kKunlun>::R_32F;
      default:
        throw std::runtime_error("Unsupported data type for Kunlun BLAS");
    }
  }

  static auto GetComputeType(DataType dtype) {
    switch (dtype) {
      case DataType::kFloat16:
      case DataType::kBFloat16:
      case DataType::kFloat32:
        return Blas<Device::Type::kKunlun>::BLAS_COMPUTE_32F;
      default:
        throw std::runtime_error("Unsupported compute type for Kunlun BLAS");
    }
  }
};

}  // namespace infini::ops

#endif
