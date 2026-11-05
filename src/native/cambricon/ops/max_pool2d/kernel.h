#ifndef INFINI_OPS_CAMBRICON_MAX_POOL2D_KERNEL_H_
#define INFINI_OPS_CAMBRICON_MAX_POOL2D_KERNEL_H_

#include "base/max_pool2d.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<MaxPool2d, Device::Type::kCambricon> : public MaxPool2d {
 public:
  Operator(const Tensor input, const std::vector<int64_t> kernel_size,
           const std::vector<int64_t> stride, const std::vector<int64_t> padding,
           const std::vector<int64_t> dilation, bool ceil_mode, Tensor out)
      : MaxPool2d{input, kernel_size, stride, padding, dilation, ceil_mode, out} {
    if (kernel_size.size() != 2 || stride.size() != 2 ||
        padding.size() != 2 || dilation.size() != 2 ||
        dilation[0] != 1 || dilation[1] != 1)
      throw std::invalid_argument("Cambricon MaxPool requires 2D parameters and unit dilation");
    const auto kh = kernel_size[0], kw = kernel_size[1];
    const auto sh = stride[0], sw = stride[1];
    const auto ph = padding[0], pw = padding[1];
    if ((input.ndim() != 3 && input.ndim() != 4) || input.ndim() != out.ndim() ||
        input.dtype() != out.dtype() ||
        (input.dtype() != DataType::kFloat16 && input.dtype() != DataType::kFloat32) ||
        input.device() != out.device() || !input.IsContiguous() ||
        !out.IsContiguous() || kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 ||
        ph < 0 || pw < 0 || (input.ndim() == 3 && (kh != 1 || ph != 0 || sh != 1)))
      throw std::invalid_argument("Unsupported Cambricon MaxPool tensor metadata");
    if (!out.numel()) return;
    handle_ = cnnl_utils::CreateHandle();
    input_ = Describe(input);
    output_ = Describe(out);
    cnnlPoolingDescriptor_t desc = nullptr;
    INFINI_OPS_CNNL_CHECK(cnnlCreatePoolingDescriptor(&desc));
    pool_.reset(desc);
    INFINI_OPS_CNNL_CHECK(cnnlSetPooling2dDescriptor_v2(
        desc, CNNL_POOLING_MAX, CNNL_PROPAGATE_NAN, kh, kw, ph, ph, pw, pw,
        sh, sw, 1, 1, ceil_mode));
    INFINI_OPS_CNNL_CHECK(cnnlGetPoolingWorkspaceSize_v2(
        handle_.get(), pool_.get(), input_.get(), output_.get(), &workspace_size_));
    scratch_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
  }

  std::size_t workspace_size_in_bytes() const override { return workspace_size_; }

  void operator()(const Tensor input, const std::vector<int64_t>,
                  const std::vector<int64_t>, const std::vector<int64_t>,
                  const std::vector<int64_t>, bool, Tensor out) const override {
    if (!pool_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    auto* scratch = workspace_ && workspace_size_in_bytes_ >= workspace_size_
                        ? workspace_ : scratch_.get();
    const float one = 1.0f, zero = 0.0f;
    INFINI_OPS_CNNL_CHECK(cnnlPoolingForward(
        handle_.get(), pool_.get(), &one, input_.get(), input.data(), &zero,
        output_.get(), out.data(), scratch, workspace_size_));
  }

 private:
  static cnnl_utils::TensorDescriptor Describe(const Tensor& tensor) {
    auto desc = cnnl_utils::CreateTensorDescriptor();
    const int dims[] = {static_cast<int>(tensor.size(0)),
                        static_cast<int>(tensor.size(1)),
                        tensor.ndim() == 4 ? static_cast<int>(tensor.size(2)) : 1,
                        static_cast<int>(tensor.size(tensor.ndim() - 1))};
    INFINI_OPS_CNNL_CHECK(cnnlSetTensorDescriptor(
        desc.get(), CNNL_LAYOUT_NCHW, cnnl_utils::GetDataType(tensor.dtype()), 4, dims));
    return desc;
  }

  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_, output_;
  std::unique_ptr<std::remove_pointer_t<cnnlPoolingDescriptor_t>,
                  decltype(&cnnlDestroyPoolingDescriptor)>
      pool_{nullptr, cnnlDestroyPoolingDescriptor};
  cnrt_utils::DeviceBuffer scratch_;
  std::size_t workspace_size_ = 0;
};

}  // namespace infini::ops

#endif
