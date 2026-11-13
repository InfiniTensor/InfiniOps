#ifndef INFINI_OPS_CAMBRICON_ACTIVATION_CNNL_H_
#define INFINI_OPS_CAMBRICON_ACTIVATION_CNNL_H_

#include <stdexcept>

#include "native/cambricon/cnnl_utils.h"

namespace infini::ops::cnnl_utils {

template <typename Base, cnnlActivationMode_t Mode>
class Activation : public Base {
 public:
  Activation(const Tensor input, Tensor out) : Base{input, out} {
    if (input.shape() != out.shape() || input.dtype() != out.dtype() ||
        input.device() != out.device() || !input.IsContiguous() ||
        !out.IsContiguous() ||
        (out.dtype() != DataType::kFloat32 && out.dtype() != DataType::kFloat16))
      throw std::invalid_argument("Cambricon activation requires matching FP32 or FP16 tensors");
    if (out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    descriptor_ = cnnl_utils::MakeTensorDescriptor(
        input.dtype(), Tensor::Shape(input.shape()), Tensor::Strides(input.strides()));
    cnnlActivationDescriptor_t value = nullptr;
    INFINI_OPS_CNNL_CHECK(cnnlCreateActivationDescriptor(&value));
    operation_.reset(value);
    INFINI_OPS_CNNL_CHECK(cnnlSetActivationDescriptor_v5(
        value, Mode, CNNL_ACTIVATION_HIGH_PRECISION,
        CNNL_PROPAGATE_NAN, 0.0f, 0.0f, 0.0f, 0.0f, true));
  }

  void operator()(const Tensor input, Tensor out) const override {
    if (!descriptor_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(this->stream_)));
    const float one = 1.0f, zero = 0.0f;
    INFINI_OPS_CNNL_CHECK(cnnlActivationForward(
        handle_.get(), operation_.get(), &one, descriptor_.get(), input.data(),
        &zero, descriptor_.get(), out.data()));
  }

 private:
  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor descriptor_;
  std::unique_ptr<std::remove_pointer_t<cnnlActivationDescriptor_t>,
                  decltype(&cnnlDestroyActivationDescriptor)>
      operation_{nullptr, cnnlDestroyActivationDescriptor};
};

}  // namespace infini::ops::cnnl_utils

#endif
