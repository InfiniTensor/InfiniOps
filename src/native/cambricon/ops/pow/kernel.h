#ifndef INFINI_OPS_CAMBRICON_POW_KERNEL_H_
#define INFINI_OPS_CAMBRICON_POW_KERNEL_H_

#include "base/pow.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<Pow, Device::Type::kCambricon> : public Pow {
 public:
  Operator(const Tensor input, const Tensor exponent, Tensor out)
      : Pow{input, exponent, out} {
    if (input.dtype() != out.dtype() || exponent.dtype() != out.dtype() ||
        input.device() != out.device() || exponent.device() != out.device() ||
        !input.IsContiguous() || !exponent.IsContiguous() || !out.IsContiguous() ||
        (out.dtype() != DataType::kFloat32 && out.dtype() != DataType::kFloat16))
      throw std::invalid_argument("Cambricon Pow requires matching FP32 or FP16 tensors");
    if (out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    lhs_ = cnnl_utils::MakeTensorDescriptor(input);
    rhs_ = cnnl_utils::MakeTensorDescriptor(exponent);
    out_ = cnnl_utils::MakeTensorDescriptor(out);
    INFINI_OPS_CNNL_CHECK(cnnlGetPowWorkspaceSize(
        handle_.get(), lhs_.get(), rhs_.get(), out_.get(), &workspace_size_));
    scratch_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
    if (workspace_size_ && !scratch_)
      throw std::runtime_error("Cambricon Pow workspace allocation failed");
  }

  Operator(const Tensor input, const double exponent, Tensor out)
      : Pow{input, exponent, out} { Unsupported(); }
  Operator(Tensor input, const Tensor exponent)
      : Operator{input, exponent, input} {}
  Operator(Tensor input, const double exponent)
      : Pow{input, exponent, input} { Unsupported(); }

  std::size_t workspace_size_in_bytes() const override { return workspace_size_; }

  void operator()(const Tensor input, const Tensor exponent, Tensor out) const override {
    if (!out_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    auto* scratch = workspace_ && workspace_size_in_bytes_ >= workspace_size_
                        ? workspace_ : scratch_.get();
    INFINI_OPS_CNNL_CHECK(cnnlPow_v2(
        handle_.get(), lhs_.get(), input.data(), rhs_.get(), exponent.data(),
        scratch, workspace_size_, out_.get(), out.data()));
  }

  void operator()(const Tensor, const double, Tensor) const override { Unsupported(); }
  void operator()(Tensor, const double) const override { Unsupported(); }
  void operator()(Tensor input, const Tensor exponent) const override {
    (*this)(input, exponent, input);
  }

 private:
  [[noreturn]] static void Unsupported() {
    throw std::invalid_argument("Cambricon Pow requires a tensor exponent");
  }


  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor lhs_, rhs_, out_;
  cnrt_utils::DeviceBuffer scratch_;
  std::size_t workspace_size_ = 0;
};

}  // namespace infini::ops

#endif
