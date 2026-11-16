#ifndef INFINI_OPS_CAMBRICON_DIV_KERNEL_H_
#define INFINI_OPS_CAMBRICON_DIV_KERNEL_H_

#include "base/div.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<Div, Device::Type::kCambricon> : public Div {
 public:
  Operator(const Tensor input, const Tensor other, Tensor out)
      : Operator{input, other, std::nullopt, out} {}

  Operator(const Tensor input, const Tensor other,
           const std::optional<std::string> rounding_mode, Tensor out)
      : Div{input, other, rounding_mode, out} {
    RequireTrueDivision(rounding_mode);
    if (input.dtype() != out.dtype() || other.dtype() != out.dtype() ||
        input.device() != out.device() || other.device() != out.device() ||
        !input.IsContiguous() || !other.IsContiguous() || !out.IsContiguous() ||
        (out.dtype() != DataType::kFloat32 && out.dtype() != DataType::kFloat16))
      throw std::invalid_argument("Cambricon Div requires matching FP32 or FP16 tensors");
    if (out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    lhs_ = Describe(input, out.ndim());
    rhs_ = Describe(other, out.ndim());
    out_ = Describe(out, out.ndim());
    INFINI_OPS_CNNL_CHECK(cnnlGetDivWorkspaceSize(
        handle_.get(), lhs_.get(), rhs_.get(), out_.get(), &workspace_size_));
    scratch_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
    if (workspace_size_ && !scratch_)
      throw std::runtime_error("Cambricon Div workspace allocation failed");
  }

  Operator(Tensor input, const Tensor other)
      : Operator{input, other, input} {}

  Operator(Tensor input, const Tensor other,
           const std::optional<std::string> rounding_mode)
      : Operator{input, other, rounding_mode, input} {}

  Operator(const Tensor input, const double other, Tensor out)
      : Div{input, other, out} {
    UnsupportedScalar();
  }

  Operator(const Tensor input, const double other,
           const std::optional<std::string> rounding_mode, Tensor out)
      : Div{input, other, rounding_mode, out} {
    UnsupportedScalar();
  }

  std::size_t workspace_size_in_bytes() const override { return workspace_size_; }

  void operator()(const Tensor input, const Tensor other, Tensor out) const override {
    if (!out_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    auto* scratch = workspace_ && workspace_size_in_bytes_ >= workspace_size_
                        ? workspace_ : scratch_.get();
    INFINI_OPS_CNNL_CHECK(cnnlDiv_v2(
        handle_.get(), CNNL_COMPUTATION_HIGH_PRECISION, lhs_.get(), input.data(),
        rhs_.get(), other.data(), scratch, workspace_size_, out_.get(), out.data()));
  }

  void operator()(const Tensor input, const Tensor other,
                  const std::optional<std::string> rounding_mode, Tensor out) const override {
    RequireTrueDivision(rounding_mode);
    (*this)(input, other, out);
  }

  void operator()(Tensor input, const Tensor other) const override {
    (*this)(input, other, input);
  }

  void operator()(Tensor input, const Tensor other,
                  const std::optional<std::string> rounding_mode) const override {
    (*this)(input, other, rounding_mode, input);
  }

  void operator()(const Tensor, const double, Tensor) const override {
    UnsupportedScalar();
  }

  void operator()(const Tensor, const double,
                  const std::optional<std::string>, Tensor) const override {
    UnsupportedScalar();
  }

 private:
  static void RequireTrueDivision(const std::optional<std::string>& rounding_mode) {
    if (rounding_mode)
      throw std::invalid_argument("Cambricon Div does not support rounding modes");
  }

  [[noreturn]] static void UnsupportedScalar() {
    throw std::invalid_argument("Cambricon Div requires a tensor divisor");
  }

  static cnnl_utils::TensorDescriptor Describe(const Tensor& tensor, size_t rank) {
    if (tensor.ndim() > rank)
      throw std::invalid_argument("Cambricon Div operand rank exceeds output rank");
    auto desc = cnnl_utils::CreateTensorDescriptor();
    const auto dims = tensor.shape();
    std::vector<int64_t> shape(rank - tensor.ndim(), 1);
    shape.insert(shape.end(), dims.begin(), dims.end());
    if (shape.empty()) shape.push_back(1);
    INFINI_OPS_CNNL_CHECK(cnnlSetTensorDescriptor_v2(
        desc.get(), CNNL_LAYOUT_ARRAY, cnnl_utils::GetDataType(tensor.dtype()),
        static_cast<int>(shape.size()), shape.data()));
    return desc;
  }

  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor lhs_, rhs_, out_;
  cnrt_utils::DeviceBuffer scratch_;
  std::size_t workspace_size_ = 0;
};

}  // namespace infini::ops

#endif
