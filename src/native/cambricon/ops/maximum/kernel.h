#ifndef INFINI_OPS_CAMBRICON_MAXIMUM_KERNEL_H_
#define INFINI_OPS_CAMBRICON_MAXIMUM_KERNEL_H_

#include "base/maximum.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<Maximum, Device::Type::kCambricon> : public Maximum {
 public:
  Operator(const Tensor input, const Tensor other, Tensor out)
      : Maximum{input, other, out} {
    if (input.dtype() != other.dtype() || input.dtype() != out.dtype() ||
        input.device() != other.device() || input.device() != out.device() ||
        !input.IsContiguous() || !other.IsContiguous() || !out.IsContiguous() ||
        (out.dtype() != DataType::kFloat32 && out.dtype() != DataType::kFloat16))
      throw std::invalid_argument("Cambricon Maximum requires matching FP32 or FP16 tensors");
    if (out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    lhs_ = cnnl_utils::MakeTensorDescriptor(input);
    rhs_ = cnnl_utils::MakeTensorDescriptor(other);
    out_ = cnnl_utils::MakeTensorDescriptor(out);
    INFINI_OPS_CNNL_CHECK(cnnlGetMaximumWorkspaceSize(handle_.get(), out_.get(), &workspace_size_));
    scratch_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
    if (workspace_size_ && !scratch_)
      throw std::runtime_error("Cambricon Maximum workspace allocation failed");
  }

  std::size_t workspace_size_in_bytes() const override { return workspace_size_; }

  void operator()(const Tensor input, const Tensor other, Tensor out) const override {
    if (!out_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    auto* scratch = workspace_ && workspace_size_in_bytes_ >= workspace_size_
                        ? workspace_ : scratch_.get();
    INFINI_OPS_CNNL_CHECK(cnnlMaximum(handle_.get(), lhs_.get(), input.data(),
        rhs_.get(), other.data(), out_.get(), out.data(), scratch, workspace_size_));
  }

 private:
  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor lhs_, rhs_, out_;
  cnrt_utils::DeviceBuffer scratch_;
  std::size_t workspace_size_ = 0;
};

}  // namespace infini::ops

#endif
