#ifndef INFINI_OPS_CAMBRICON_ROI_ALIGN_KERNEL_H_
#define INFINI_OPS_CAMBRICON_ROI_ALIGN_KERNEL_H_

#include <cmath>
#include <limits>

#include "base/roi_align.h"
#include "native/cambricon/cnnl_utils.h"

namespace infini::ops {

template <>
class Operator<RoiAlign, Device::Type::kCambricon> : public RoiAlign {
 public:
  Operator(const Tensor input, const Tensor boxes, int64_t sampling_ratio,
           float spatial_scale, bool aligned, Tensor out)
      : RoiAlign{input, boxes, sampling_ratio, spatial_scale, aligned, out} {
    if (input.ndim() != 4 || boxes.ndim() != 2 || out.ndim() != 4 ||
        boxes.size(1) != 5 || out.size(0) != boxes.size(0) ||
        out.size(3) != input.size(3) || out.size(1) == 0 || out.size(2) == 0 ||
        out.size(1) > std::numeric_limits<int>::max() ||
        out.size(2) > std::numeric_limits<int>::max() ||
        sampling_ratio < 0 || sampling_ratio > std::numeric_limits<int>::max() ||
        !std::isfinite(spatial_scale) || spatial_scale <= 0 ||
        input.device() != out.device() || boxes.device() != out.device() ||
        input.dtype() != out.dtype() || boxes.dtype() != out.dtype() ||
        (out.dtype() != DataType::kFloat32 && out.dtype() != DataType::kFloat16) ||
        !input.IsContiguous() || !boxes.IsContiguous() || !out.IsContiguous())
      throw std::invalid_argument("Unsupported Cambricon RoiAlign parameters");
    if (out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    input_ = Describe(input, CNNL_LAYOUT_NHWC);
    boxes_ = Describe(boxes, CNNL_LAYOUT_ARRAY);
    out_ = Describe(out, CNNL_LAYOUT_NHWC);
    cnnlRoiAlignDescriptor_t desc = nullptr;
    INFINI_OPS_CNNL_CHECK(cnnlCreateRoiAlignDescriptor(&desc));
    roi_.reset(desc);
    INFINI_OPS_CNNL_CHECK(cnnlSetRoiAlignDescriptor_v2(
        desc, out.size(1), out.size(2), sampling_ratio, spatial_scale, 1, aligned));
  }

  void operator()(const Tensor input, const Tensor boxes, int64_t, float, bool,
                  Tensor out) const override {
    if (!out_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    INFINI_OPS_CNNL_CHECK(cnnlRoiAlign_v2(
        handle_.get(), roi_.get(), input_.get(), input.data(), boxes_.get(),
        boxes.data(), out_.get(), out.data(), nullptr, nullptr, nullptr, nullptr));
  }

 private:
  static cnnl_utils::TensorDescriptor Describe(const Tensor& tensor,
                                             cnnlTensorLayout_t layout) {
    auto desc = cnnl_utils::CreateTensorDescriptor();
    const auto shape = tensor.shape();
    const std::vector<int64_t> dims(shape.begin(), shape.end());
    INFINI_OPS_CNNL_CHECK(cnnlSetTensorDescriptor_v2(
        desc.get(), layout, cnnl_utils::GetDataType(tensor.dtype()),
        static_cast<int>(dims.size()), dims.data()));
    return desc;
  }

  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_, boxes_, out_;
  std::unique_ptr<std::remove_pointer_t<cnnlRoiAlignDescriptor_t>,
                  decltype(&cnnlDestroyRoiAlignDescriptor)>
      roi_{nullptr, cnnlDestroyRoiAlignDescriptor};
};

}  // namespace infini::ops

#endif
