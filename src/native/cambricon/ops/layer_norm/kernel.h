#ifndef INFINI_OPS_CAMBRICON_LAYER_NORM_KERNEL_H_
#define INFINI_OPS_CAMBRICON_LAYER_NORM_KERNEL_H_

#include <cmath>
#include <stdexcept>

#include "base/layer_norm.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<LayerNorm, Device::Type::kCambricon> : public LayerNorm {
 public:
  Operator(const Tensor input, const Tensor scale, std::optional<Tensor> bias,
           float eps, Tensor out)
      : LayerNorm{input, scale, bias, eps, out}, eps_(eps) {
    const auto rank = static_cast<int64_t>(input.ndim());
    const auto axis = rank - 1;
    if (axis < 0 || axis >= rank || !std::isfinite(eps) || eps < 0 ||
        input.shape() != out.shape() ||
        (input.dtype() != DataType::kFloat32 && input.dtype() != DataType::kFloat16) ||
        !Compatible(out, input) || !Compatible(scale, input) ||
        !input.IsContiguous() || (bias && (!Compatible(*bias, input) || bias->shape() != scale.shape())))
      throw std::invalid_argument("Invalid Cambricon LayerNorm tensor metadata");
    axis_ = static_cast<int>(axis);
    Tensor::Shape prefix, suffix;
    for (int64_t i = 0; i < rank; ++i) {
      if (i < axis) prefix.push_back(input.shape()[i]);
      else suffix.push_back(input.shape()[i]);
    }
    if (Tensor::Shape(scale.shape()) != suffix)
      throw std::invalid_argument("Cambricon LayerNorm scale must match the normalized shape");
    if (out.numel() == 0) return;
    size_t stats_size = 1;
    for (auto dim : prefix) stats_size *= dim;
    Tensor::Strides stats_strides(prefix.size());
    Tensor::Stride stride = 1;
    for (size_t i = prefix.size(); i-- > 0;) {
      stats_strides[i] = stride;
      stride *= prefix[i];
    }
    handle_ = cnnl_utils::CreateHandle();
    input_ = cnnl_utils::MakeTensorDescriptor(input);
    scale_ = cnnl_utils::MakeTensorDescriptor(scale);
    output_ = cnnl_utils::MakeTensorDescriptor(out);
    stats_ = cnnl_utils::MakeTensorDescriptor(input.dtype(), prefix, stats_strides);
    INFINI_OPS_CNNL_CHECK(cnnlGetLayerNormOpWorkspaceSize(
        handle_.get(), axis_, input_.get(), &workspace_size_));
    workspace_own_ = Allocate(workspace_size_);
    const auto stats_bytes = stats_size * ::infini::rt::kDataTypeToSize.at(input.dtype());
    mean_ = Allocate(stats_bytes);
    rstd_ = Allocate(stats_bytes);
  }

  std::size_t workspace_size_in_bytes() const override { return workspace_size_; }

  void operator()(const Tensor input, const Tensor scale, std::optional<Tensor> bias,
                  float, Tensor out) const override {
    if (!output_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    auto* scratch = workspace_ && workspace_size_in_bytes_ >= workspace_size_
                        ? workspace_ : workspace_own_.get();
    INFINI_OPS_CNNL_CHECK(cnnlLayerNormForward(
        handle_.get(), input_.get(), input.data(), axis_, scale_.get(), scale.data(),
        bias ? bias->data() : nullptr, eps_, scratch, workspace_size_,
        output_.get(), out.data(), stats_.get(), mean_.get(), rstd_.get()));
  }

 private:
  static bool Compatible(const Tensor& tensor, const Tensor& input) {
    return tensor.dtype() == input.dtype() && tensor.device() == input.device() &&
           tensor.IsContiguous();
  }
  static cnrt_utils::DeviceBuffer Allocate(std::size_t bytes) {
    auto result = cnrt_utils::AllocateDeviceBuffer(bytes);
    if (bytes && !result) throw std::runtime_error("Cambricon LayerNorm allocation failed");
    return result;
  }
  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_, scale_, output_, stats_;
  cnrt_utils::DeviceBuffer workspace_own_, mean_, rstd_;
  std::size_t workspace_size_ = 0;
  int axis_ = 0;
  float eps_;
};

}  // namespace infini::ops

#endif
