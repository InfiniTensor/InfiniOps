#ifndef INFINI_OPS_CAMBRICON_COMPARE_CNNL_H_
#define INFINI_OPS_CAMBRICON_COMPARE_CNNL_H_

#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops::cnnl_utils {

template <typename Base, cnnlLogicOp_t Kind>
class Compare : public Base {
 public:
  Compare(const Tensor input, const Tensor other, Tensor out)
      : Base{input, other, out} {
    if (input.dtype() != other.dtype() || out.dtype() != DataType::kBool ||
        input.device() != out.device() || other.device() != out.device() ||
        !input.IsContiguous() || !other.IsContiguous() || !out.IsContiguous())
      throw std::invalid_argument("Cambricon comparison requires matching inputs and BOOL output");
    if (out.numel() == 0) return;
    handle_ = CreateHandle();
    lhs_ = MakeTensorDescriptor(input);
    rhs_ = MakeTensorDescriptor(other);
    out_ = MakeTensorDescriptor(out);
    INFINI_OPS_CNNL_CHECK(cnnlGetLogicOpWorkspaceSize(
        handle_.get(), lhs_.get(), rhs_.get(), out_.get(), &workspace_size_));
    scratch_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
    if (workspace_size_ && !scratch_)
      throw std::runtime_error("Cambricon comparison workspace allocation failed");
  }
  Compare(const Tensor input, const double other, Tensor out)
      : Base{input, other, out} { Unsupported(); }
  Compare(Tensor input, const Tensor other)
      : Base{input, other, input} { Unsupported(); }
  Compare(Tensor input, const double other)
      : Base{input, other, input} { Unsupported(); }

  std::size_t workspace_size_in_bytes() const override { return workspace_size_; }

  void operator()(const Tensor input, const Tensor other, Tensor out) const override {
    if (!out_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(this->stream_)));
    auto* scratch = this->workspace_ && this->workspace_size_in_bytes_ >= workspace_size_
                        ? this->workspace_ : scratch_.get();
    INFINI_OPS_CNNL_CHECK(cnnlLogicOp(
        handle_.get(), Kind, lhs_.get(), input.data(), rhs_.get(), other.data(),
        scratch, workspace_size_, out_.get(), out.data()));
  }
  void operator()(const Tensor, const double, Tensor) const override { Unsupported(); }
  void operator()(Tensor, const Tensor) const override { Unsupported(); }
  void operator()(Tensor, const double) const override { Unsupported(); }

 private:
  [[noreturn]] static void Unsupported() {
    throw std::invalid_argument("Cambricon comparison requires two tensors and a separate BOOL output");
  }

  Handle handle_;
  TensorDescriptor lhs_, rhs_, out_;
  cnrt_utils::DeviceBuffer scratch_;
  std::size_t workspace_size_ = 0;
};

}  // namespace infini::ops::cnnl_utils

#endif
