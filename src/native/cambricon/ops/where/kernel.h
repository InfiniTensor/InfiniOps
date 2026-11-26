#ifndef INFINI_OPS_CAMBRICON_WHERE_KERNEL_H_
#define INFINI_OPS_CAMBRICON_WHERE_KERNEL_H_

#include <stdexcept>

#include "base/where.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<Where, Device::Type::kCambricon> : public Where {
 public:
  Operator(const Tensor condition, const Tensor input, const Tensor other,
           Tensor out) : Where{condition, input, other, out} {
    if (condition_type_ != DataType::kBool || input_type_ != out_type_ ||
        other_type_ != out_type_ || condition.device() != out.device() ||
        input.device() != out.device() || other.device() != out.device() ||
        !condition.IsContiguous() || !input.IsContiguous() ||
        !other.IsContiguous() || !out.IsContiguous()) {
      throw std::invalid_argument("Cambricon Where requires a Boolean mask and contiguous tensors");
    }
    for (const auto& value : {condition, input, other}) {
      if (value.ndim() > out.ndim())
        throw std::invalid_argument("Cambricon Where: invalid broadcast rank");
      const auto offset = out.ndim() - value.ndim();
      for (size_t i = 0; i < value.ndim(); ++i) {
        if (value.shape()[i] != 1 && value.shape()[i] != out.shape()[offset + i])
          throw std::invalid_argument("Cambricon Where: invalid broadcast dimensions");
      }
    }
    if (out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    condition_ = cnnl_utils::MakeTensorDescriptor(condition_type_, condition_shape_, condition_strides_);
    input_ = cnnl_utils::MakeTensorDescriptor(input_type_, input_shape_, input_strides_);
    other_ = cnnl_utils::MakeTensorDescriptor(other_type_, other_shape_, other_strides_);
    output_ = cnnl_utils::MakeTensorDescriptor(out_type_, out_shape_, out_strides_);
    INFINI_OPS_CNNL_CHECK(cnnlGetSelectV2WorkspaceSize(
        handle_.get(), condition_.get(), input_.get(), other_.get(), &workspace_size_));
    workspace_storage_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
    if (workspace_size_ && !workspace_storage_)
      throw std::runtime_error("Cambricon Where: workspace allocation failed");
  }

  std::size_t workspace_size_in_bytes() const override { return workspace_size_; }

  void operator()(const Tensor condition, const Tensor input, const Tensor other,
                  Tensor out) const override {
    if (!output_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    auto* scratch = workspace_ && workspace_size_in_bytes_ >= workspace_size_
                        ? workspace_ : workspace_storage_.get();
    INFINI_OPS_CNNL_CHECK(cnnlSelectV2(
        handle_.get(), condition_.get(), condition.data(), input_.get(), input.data(),
        other_.get(), other.data(), scratch, workspace_size_, output_.get(), out.data()));
  }

 private:
  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor condition_, input_, other_, output_;
  cnrt_utils::DeviceBuffer workspace_storage_;
  std::size_t workspace_size_ = 0;
};

}  // namespace infini::ops

#endif
