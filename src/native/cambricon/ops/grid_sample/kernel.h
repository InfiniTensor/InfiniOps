#ifndef INFINI_OPS_CAMBRICON_GRID_SAMPLE_KERNEL_H_
#define INFINI_OPS_CAMBRICON_GRID_SAMPLE_KERNEL_H_

#include "base/grid_sample.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<GridSample, Device::Type::kCambricon> : public GridSample {
 public:
  Operator(const Tensor input, const Tensor grid, bool nearest,
           bool reflection, bool align_corners, Tensor out)
      : GridSample{input, grid, nearest, reflection, align_corners, out} {
    if (input.ndim() != 4 || grid.ndim() != 4 || out.ndim() != 4 ||
        input.dtype() != out.dtype() || grid.dtype() != out.dtype() ||
        (out.dtype() != DataType::kFloat32 && out.dtype() != DataType::kFloat16) ||
        input.device() != out.device() || grid.device() != out.device() ||
        !input.IsContiguous() || !grid.IsContiguous() || !out.IsContiguous() ||
        grid.size(3) != 2 || input.size(0) != grid.size(0) ||
        out.size(0) != grid.size(0) || out.size(1) != grid.size(1) ||
        out.size(2) != grid.size(2) || out.size(3) != input.size(3) ||
        (nearest && reflection))
      throw std::invalid_argument("Unsupported Cambricon GridSample parameters");
    if (out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    input_ = Describe(input);
    grid_ = Describe(grid);
    out_ = Describe(out);
    cnnlGridSampleDescriptor_t desc = nullptr;
    INFINI_OPS_CNNL_CHECK(cnnlCreateGridSampleDescriptor(&desc));
    sample_.reset(desc);
    INFINI_OPS_CNNL_CHECK(cnnlSetGridSampleDescriptor(
        desc, nearest ? CNNL_INTERP_NEAREST : CNNL_INTERP_BILINEAR,
        reflection ? CNNL_GRIDSAMPLE_PADDING_REFLECTION : CNNL_GRIDSAMPLE_PADDING_ZEROS,
        align_corners));
    INFINI_OPS_CNNL_CHECK(cnnlGetGridSampleForwardWorkspaceSize(
        handle_.get(), input_.get(), grid_.get(), out_.get(), &workspace_size_));
    scratch_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
    if (workspace_size_ && !scratch_)
      throw std::runtime_error("Cambricon GridSample workspace allocation failed");
  }

  std::size_t workspace_size_in_bytes() const override { return workspace_size_; }

  void operator()(const Tensor input, const Tensor grid, bool, bool, bool,
                  Tensor out) const override {
    if (!out_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    auto* scratch = workspace_ && workspace_size_in_bytes_ >= workspace_size_
                        ? workspace_ : scratch_.get();
    INFINI_OPS_CNNL_CHECK(cnnlGridSampleForward(
        handle_.get(), sample_.get(), input_.get(), input.data(),
        grid_.get(), grid.data(), out_.get(), out.data(), scratch, workspace_size_));
  }

 private:
  static cnnl_utils::TensorDescriptor Describe(const Tensor& tensor) {
    auto desc = cnnl_utils::CreateTensorDescriptor();
    const auto shape = tensor.shape();
    const std::vector<int64_t> dims(shape.begin(), shape.end());
    INFINI_OPS_CNNL_CHECK(cnnlSetTensorDescriptor_v2(
        desc.get(), CNNL_LAYOUT_NHWC, cnnl_utils::GetDataType(tensor.dtype()),
        static_cast<int>(dims.size()), dims.data()));
    return desc;
  }

  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_, grid_, out_;
  std::unique_ptr<std::remove_pointer_t<cnnlGridSampleDescriptor_t>,
                  decltype(&cnnlDestroyGridSampleDescriptor)>
      sample_{nullptr, cnnlDestroyGridSampleDescriptor};
  cnrt_utils::DeviceBuffer scratch_;
  std::size_t workspace_size_ = 0;
};

}  // namespace infini::ops

#endif
