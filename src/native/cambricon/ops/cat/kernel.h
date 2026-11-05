#ifndef INFINI_OPS_CAMBRICON_CAT_KERNEL_H_
#define INFINI_OPS_CAMBRICON_CAT_KERNEL_H_

#include <limits>
#include <stdexcept>

#include "base/cat.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<Cat, Device::Type::kCambricon> : public Cat {
 public:
  Operator(const std::vector<Tensor> tensors, int64_t dim, Tensor out)
      : Cat{tensors, CheckedAxis(tensors, dim, out), out} {
    if (out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    output_ = cnnl_utils::MakeTensorDescriptor(out_type_, out_shape_, out_strides_);
    for (size_t i = 0; i < tensors.size(); ++i) {
      const auto& input = tensors[i];
      if (input.numel() == 0) continue;
      indices_.push_back(i);
      inputs_.push_back(cnnl_utils::MakeTensorDescriptor(
          input.dtype(), Tensor::Shape(input.shape()), Tensor::Strides(input.strides())));
      descriptors_.push_back(inputs_.back().get());
    }
    if (indices_.size() == 1) return;
    INFINI_OPS_CNNL_CHECK(cnnlGetConcatWorkspaceSize(
        handle_.get(), indices_.size(), &workspace_size_));
    workspace_own_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
    if (workspace_size_ && !workspace_own_)
      throw std::runtime_error("Cambricon Cat workspace allocation failed");
  }

  std::size_t workspace_size_in_bytes() const override { return workspace_size_; }

  void operator()(const std::vector<Tensor> tensors, int64_t, Tensor out) const override {
    if (!output_) return;
    if (indices_.size() == 1) {
      const auto* data = tensors[indices_.front()].data();
      if (data != out.data() &&
          cnrtMemcpyAsync(out.data(), const_cast<void*>(data), out.numel() * out.element_size(),
                          static_cast<cnrtQueue_t>(stream_), cnrtMemcpyDevToDev) != cnrtSuccess)
        throw std::runtime_error("Cambricon Cat single-input copy failed");
      return;
    }
    std::vector<const void*> buffers;
    buffers.reserve(indices_.size());
    for (auto index : indices_) buffers.push_back(tensors[index].data());
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    auto* scratch = workspace_ && workspace_size_in_bytes_ >= workspace_size_
                        ? workspace_ : workspace_own_.get();
    INFINI_OPS_CNNL_CHECK(cnnlConcat(
        handle_.get(), buffers.size(), dim_, descriptors_.data(), buffers.data(),
        scratch, workspace_size_, output_.get(), out.data()));
  }

 private:
  static int64_t CheckedAxis(const std::vector<Tensor>& tensors, int64_t axis, const Tensor& out) {
    const auto rank = static_cast<int64_t>(out.ndim());
    if (axis < 0) axis += rank;
    if (tensors.empty() || tensors.size() > std::numeric_limits<int>::max() ||
        axis < 0 || axis >= rank || !out.IsContiguous())
      throw std::invalid_argument("Invalid Cambricon Cat input list or axis");
    size_t extent = 0;
    for (const auto& tensor : tensors) {
      if (tensor.ndim() != out.ndim())
        throw std::invalid_argument("Cambricon Cat input rank differs from output");
      if (tensor.dtype() != out.dtype() || tensor.device() != out.device())
        throw std::invalid_argument("Cambricon Cat input dtype or device differs from output");
      if (tensor.numel() != 0 && !tensor.IsContiguous())
        throw std::invalid_argument("Cambricon Cat requires contiguous nonempty inputs");
      for (int64_t i = 0; i < rank; ++i) {
        if (i != axis && tensor.size(i) != out.size(i))
          throw std::invalid_argument("Inconsistent Cambricon Cat dimensions");
      }
      extent += tensor.size(axis);
    }
    if (extent != out.size(axis))
      throw std::invalid_argument("Invalid Cambricon Cat output shape");
    return axis;
  }

  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor output_;
  std::vector<cnnl_utils::TensorDescriptor> inputs_;
  std::vector<cnnlTensorDescriptor_t> descriptors_;
  std::vector<size_t> indices_;
  cnrt_utils::DeviceBuffer workspace_own_;
  std::size_t workspace_size_ = 0;
};

}  // namespace infini::ops

#endif
