#ifndef INFINI_OPS_CAMBRICON_CAST_KERNEL_H_
#define INFINI_OPS_CAMBRICON_CAST_KERNEL_H_

#include <stdexcept>

#include "base/cast.h"
#include "native/cambricon/cnnl_utils.h"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kCambricon> : public Cast {
 public:
  Operator(const Tensor input, Tensor out) : Cast{input, out} {
    if (input_shape_ != out_shape_ || input.device() != out.device() ||
        !is_input_contiguous_ || !is_out_contiguous_)
      throw std::invalid_argument("Cambricon Cast requires matching contiguous tensor shapes");
    cast_type_ = CastType(input_dtype_, out_dtype_);
    if (output_size_ == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    input_ = cnnl_utils::MakeTensorDescriptor(input_dtype_, input_shape_, input_strides_);
    output_ = cnnl_utils::MakeTensorDescriptor(out_dtype_, out_shape_, out_strides_);
  }

  void operator()(const Tensor input, Tensor out) const override {
    if (!output_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    INFINI_OPS_CNNL_CHECK(cnnlCastDataType(
        handle_.get(), input_.get(), input.data(), cast_type_, output_.get(), out.data()));
  }

 private:
  static cnnlCastDataType_t CastType(DataType input, DataType out) {
    struct Conversion { DataType src, dst; cnnlCastDataType_t cnnl; };
    static constexpr Conversion conversions[] = {
      {DataType::kFloat32, DataType::kFloat16, CNNL_CAST_FLOAT_TO_HALF},
      {DataType::kFloat32, DataType::kInt64, CNNL_CAST_FLOAT_TO_INT64},
      {DataType::kFloat32, DataType::kInt32, CNNL_CAST_FLOAT_TO_INT32},
      {DataType::kFloat32, DataType::kInt16, CNNL_CAST_FLOAT_TO_INT16},
      {DataType::kFloat32, DataType::kInt8, CNNL_CAST_FLOAT_TO_INT8},
      {DataType::kFloat32, DataType::kBFloat16, CNNL_CAST_FLOAT_TO_BFLOAT16},
      {DataType::kInt32, DataType::kFloat32, CNNL_CAST_INT32_TO_FLOAT},
      {DataType::kInt32, DataType::kInt8, CNNL_CAST_INT32_TO_INT8},
      {DataType::kInt32, DataType::kInt16, CNNL_CAST_INT32_TO_INT16},
      {DataType::kInt32, DataType::kInt64, CNNL_CAST_INT32_TO_INT64},
      {DataType::kInt16, DataType::kFloat32, CNNL_CAST_INT16_TO_FLOAT},
      {DataType::kInt16, DataType::kInt32, CNNL_CAST_INT16_TO_INT32},
      {DataType::kInt8, DataType::kFloat32, CNNL_CAST_INT8_TO_FLOAT},
      {DataType::kInt8, DataType::kInt16, CNNL_CAST_INT8_TO_INT16},
      {DataType::kInt8, DataType::kInt32, CNNL_CAST_INT8_TO_INT32},
      {DataType::kUInt8, DataType::kFloat32, CNNL_CAST_UINT8_TO_FLOAT},
      {DataType::kUInt8, DataType::kInt32, CNNL_CAST_UINT8_TO_INT32},
      {DataType::kUInt8, DataType::kInt64, CNNL_CAST_UINT8_TO_INT64},
      {DataType::kInt64, DataType::kInt32, CNNL_CAST_INT64_TO_INT32},
      {DataType::kInt64, DataType::kUInt32, CNNL_CAST_INT64_TO_UINT32},
      {DataType::kInt64, DataType::kFloat32, CNNL_CAST_INT64_TO_FLOAT},
      {DataType::kUInt32, DataType::kInt64, CNNL_CAST_UINT32_TO_INT64},
      {DataType::kFloat16, DataType::kFloat32, CNNL_CAST_HALF_TO_FLOAT},
      {DataType::kBFloat16, DataType::kFloat32, CNNL_CAST_BFLOAT16_TO_FLOAT},
      {DataType::kFloat32, DataType::kBool, CNNL_CAST_FLOAT_TO_BOOL},
      {DataType::kFloat16, DataType::kBool, CNNL_CAST_HALF_TO_BOOL},
      {DataType::kBFloat16, DataType::kBool, CNNL_CAST_BFLOAT16_TO_BOOL},
      {DataType::kInt32, DataType::kBool, CNNL_CAST_INT32_TO_BOOL},
      {DataType::kInt64, DataType::kBool, CNNL_CAST_INT64_TO_BOOL},
      {DataType::kBool, DataType::kFloat32, CNNL_CAST_BOOL_TO_FLOAT},
      {DataType::kBool, DataType::kFloat16, CNNL_CAST_BOOL_TO_HALF},
      {DataType::kBool, DataType::kBFloat16, CNNL_CAST_BOOL_TO_BFLOAT16},
      {DataType::kBool, DataType::kInt32, CNNL_CAST_BOOL_TO_INT32},
      {DataType::kBool, DataType::kInt64, CNNL_CAST_BOOL_TO_INT64},
    };
    for (const auto& entry : conversions) {
      if (entry.src == input && entry.dst == out)
        return entry.cnnl;
    }
    throw std::invalid_argument("Unsupported Cambricon Cast dtype pair");
  }
  cnnlCastDataType_t cast_type_{};
  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_, output_;
};

}  // namespace infini::ops

#endif
