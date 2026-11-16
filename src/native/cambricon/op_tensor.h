#ifndef INFINI_OPS_CAMBRICON_OP_TENSOR_H_
#define INFINI_OPS_CAMBRICON_OP_TENSOR_H_

#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops::cnnl_utils {

class OpTensor {
 public:
  OpTensor(cnnlOpTensorDesc_t kind, const Tensor& lhs, const Tensor& rhs,
           const Tensor& out)
      : handle_(CreateHandle()),
        lhs_(MakeTensorDescriptor(lhs)), rhs_(MakeTensorDescriptor(rhs)), out_(MakeTensorDescriptor(out)) {
    cnnlOpTensorDescriptor_t desc = nullptr;
    INFINI_OPS_CNNL_CHECK(cnnlCreateOpTensorDescriptor(&desc));
    operation_.reset(desc);
    INFINI_OPS_CNNL_CHECK(cnnlSetOpTensorDescriptor(
        desc, kind, GetDataType(out.dtype()), CNNL_PROPAGATE_NAN));
    INFINI_OPS_CNNL_CHECK(cnnlGetOpTensorWorkspaceSize(
        handle_.get(), lhs_.get(), rhs_.get(), out_.get(), &workspace_size_));
    scratch_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
    if (workspace_size_ && !scratch_)
      throw std::runtime_error("Cambricon OpTensor workspace allocation failed");
  }

  size_t WorkspaceSize() const { return workspace_size_; }

  void Run(const Tensor& lhs, const Tensor& rhs, Tensor out, float alpha,
           cnrtQueue_t queue, void* workspace, size_t capacity) const {
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), queue));
    auto* scratch = workspace && capacity >= workspace_size_
                        ? workspace : scratch_.get();
    const float one = 1.0f, zero = 0.0f;
    INFINI_OPS_CNNL_CHECK(cnnlOpTensor(
        handle_.get(), operation_.get(), &one, lhs_.get(), lhs.data(), &alpha,
        rhs_.get(), rhs.data(), scratch, workspace_size_, &zero, out_.get(),
        out.data()));
  }

 private:

  Handle handle_;
  TensorDescriptor lhs_, rhs_, out_;
  std::unique_ptr<std::remove_pointer_t<cnnlOpTensorDescriptor_t>,
                  decltype(&cnnlDestroyOpTensorDescriptor)>
      operation_{nullptr, cnnlDestroyOpTensorDescriptor};
  cnrt_utils::DeviceBuffer scratch_;
  size_t workspace_size_ = 0;
};

}  // namespace infini::ops::cnnl_utils

#endif
