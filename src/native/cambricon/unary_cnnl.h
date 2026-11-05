#ifndef INFINI_OPS_CAMBRICON_UNARY_CNNL_H_
#define INFINI_OPS_CAMBRICON_UNARY_CNNL_H_

#include "native/cambricon/cnnl_utils.h"

namespace infini::ops::cnnl_utils {

template <typename Base, auto Execute, bool Boolean = false>
class Unary : public Base {
 public:
  Unary(const Tensor input, Tensor out) : Base{input, out} {
    if (input.dtype() != out.dtype() || input.device() != out.device() ||
        input.shape() != out.shape() ||
        (Boolean ? out.dtype() != DataType::kBool
                 : out.dtype() != DataType::kFloat32 && out.dtype() != DataType::kFloat16))
      throw std::invalid_argument("Cambricon unary operation received unsupported tensor metadata");
    if (out.numel() == 0) return;
    handle_ = CreateHandle();
    input_ = MakeTensorDescriptor(
        input.dtype(), Tensor::Shape(input.shape()), Tensor::Strides(input.strides()));
    out_ = MakeTensorDescriptor(
        out.dtype(), Tensor::Shape(out.shape()), Tensor::Strides(out.strides()));
  }

  void operator()(const Tensor input, Tensor out) const override {
    if (!out_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(this->stream_)));
    INFINI_OPS_CNNL_CHECK(Execute(handle_.get(), input_.get(), input.data(),
                                 out_.get(), out.data()));
  }

 private:
  Handle handle_;
  TensorDescriptor input_, out_;
};

}  // namespace infini::ops::cnnl_utils

#endif
