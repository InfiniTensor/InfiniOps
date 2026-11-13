#ifndef INFINI_OPS_CAMBRICON_RESIZE2D_KERNEL_H_
#define INFINI_OPS_CAMBRICON_RESIZE2D_KERNEL_H_

#include <array>
#include <cmath>

#include "base/resize2d.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<Resize2d, Device::Type::kCambricon> : public Resize2d {
 public:
  Operator(const Tensor input, int64_t mode, int64_t coordinates,
           int64_t rounding, float height_scale, float width_scale, Tensor out)
      : Resize2d{input, mode, coordinates, rounding, height_scale, width_scale, out} {
    if (input.ndim() != 4 || out.ndim() != 4 || input.dtype() != out.dtype() ||
        (out.dtype() != DataType::kFloat32 && out.dtype() != DataType::kFloat16) ||
        input.device() != out.device() || !input.IsContiguous() || !out.IsContiguous() ||
        input.size(0) != out.size(0) || input.size(1) != out.size(1) ||
        input.size(2) == 0 || input.size(3) == 0 ||
        mode < kNearest || mode > kLinear || coordinates < kHalfPixel ||
        coordinates > kAsymmetric || rounding < kPreferFloor || rounding > kCeil ||
        !std::isfinite(height_scale) || !std::isfinite(width_scale) ||
        height_scale < 0 || width_scale < 0 || ((height_scale == 0) != (width_scale == 0)))
      throw std::invalid_argument("Unsupported Cambricon Resize2d parameters");
    if (out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    nearest2x_ = mode == kNearest && coordinates == kAsymmetric && rounding == kFloor &&
        (height_scale == 0 || (height_scale == 2 && width_scale == 2)) &&
        out.size(2) == 2 * input.size(2) && out.size(3) == 2 * input.size(3);
    if (nearest2x_) {
      const auto n = static_cast<int64_t>(input.size(0)), c = static_cast<int64_t>(input.size(1));
      const auto h = static_cast<int64_t>(input.size(2)), w = static_cast<int64_t>(input.size(3));
      input_ = DescribeArray(input.dtype(), {n, c, h, w, 1});
      input_nhwc_ = DescribeArray(input.dtype(), {n, c, h, w, 2});
      out_nhwc_ = DescribeArray(input.dtype(), {n, c, h, 1, w * 2});
      out_ = DescribeArray(input.dtype(), {n, c, h, 2, w * 2});
      input_buffer_ = Allocate(input.numel() * input.element_size() * 2);
      return;
    }
    static constexpr std::array<cnnlInterpAlgo_t, 4> coordinates_map{
        CNNL_INTERP_ALGO_9, CNNL_INTERP_ALGO_0, CNNL_INTERP_ALGO_2, CNNL_INTERP_ALGO_8};
    static constexpr std::array<cnnlInterpRoundMode_t, 4> rounding_map{
        CNNL_INTERP_ROUND_PERFER_FLOOR, CNNL_INTERP_ROUND_PERFER_CEIL,
        CNNL_INTERP_FLOOR, CNNL_INTERP_CEIL};
    input_ = Describe(input, false);
    input_nhwc_ = Describe(input, true);
    out_ = Describe(out, false);
    out_nhwc_ = Describe(out, true);
    SetupTranspose(to_nhwc_, {0, 2, 3, 1});
    SetupTranspose(to_nchw_, {0, 3, 1, 2});
    cnnlInterpDescriptor_t interp = nullptr;
    INFINI_OPS_CNNL_CHECK(cnnlCreateInterpDescriptor(&interp));
    interp_.reset(interp);
    const double scales[2] = {height_scale, width_scale};
    INFINI_OPS_CNNL_CHECK(cnnlSetInterpDescriptor_v2(
        interp, input_nhwc_.get(), mode == kLinear ? CNNL_INTERP_BILINEAR : CNNL_INTERP_NEAREST,
        coordinates_map[coordinates], height_scale > 0 ? scales : nullptr));
    INFINI_OPS_CNNL_CHECK(cnnlSetInterpDescriptorEx(
        interp, input_nhwc_.get(), rounding_map[rounding], nullptr, nullptr, -0.75f, false));
    INFINI_OPS_CNNL_CHECK(cnnlGetTransposeWorkspaceSize(
        handle_.get(), input_.get(), to_nhwc_.get(), &input_workspace_));
    INFINI_OPS_CNNL_CHECK(cnnlGetTransposeWorkspaceSize(
        handle_.get(), out_nhwc_.get(), to_nchw_.get(), &output_workspace_));
    scratch_size_ = std::max(input_workspace_, output_workspace_);
    input_buffer_ = Allocate(input.numel() * input.element_size());
    output_buffer_ = Allocate(out.numel() * out.element_size());
    scratch_ = Allocate(scratch_size_);
  }

  std::size_t workspace_size_in_bytes() const override { return scratch_size_; }

  void operator()(const Tensor input, int64_t, int64_t, int64_t, float, float,
                  Tensor out) const override {
    if (!out_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    if (nearest2x_) {
      INFINI_OPS_CNNL_CHECK(cnnlTile(handle_.get(), input_.get(), input.data(),
                                    input_nhwc_.get(), input_buffer_.get()));
      INFINI_OPS_CNNL_CHECK(cnnlTile(handle_.get(), out_nhwc_.get(), input_buffer_.get(),
                                    out_.get(), out.data()));
      return;
    }
    auto* scratch = workspace_ && workspace_size_in_bytes_ >= scratch_size_
                        ? workspace_ : scratch_.get();
    INFINI_OPS_CNNL_CHECK(cnnlTranspose_v2(
        handle_.get(), to_nhwc_.get(), input_.get(), input.data(),
        input_nhwc_.get(), input_buffer_.get(), scratch, input_workspace_));
    INFINI_OPS_CNNL_CHECK(cnnlInterp_v3(
        handle_.get(), interp_.get(), input_nhwc_.get(), input_buffer_.get(),
        out_nhwc_.get(), output_buffer_.get()));
    INFINI_OPS_CNNL_CHECK(cnnlTranspose_v2(
        handle_.get(), to_nchw_.get(), out_nhwc_.get(), output_buffer_.get(),
        out_.get(), out.data(), scratch, output_workspace_));
  }

 private:
  using Transpose = std::unique_ptr<std::remove_pointer_t<cnnlTransposeDescriptor_t>,
                                    decltype(&cnnlDestroyTransposeDescriptor)>;
  static cnnl_utils::TensorDescriptor DescribeArray(
      DataType dtype, std::initializer_list<int64_t> dims) {
    auto desc = cnnl_utils::CreateTensorDescriptor();
    INFINI_OPS_CNNL_CHECK(cnnlSetTensorDescriptor_v2(
        desc.get(), CNNL_LAYOUT_ARRAY, cnnl_utils::GetDataType(dtype), dims.size(), dims.begin()));
    return desc;
  }
  static cnnl_utils::TensorDescriptor Describe(const Tensor& tensor, bool nhwc) {
    auto desc = cnnl_utils::CreateTensorDescriptor();
    std::array<int64_t, 4> dims{};
    if (nhwc)
      dims = {static_cast<int64_t>(tensor.size(0)), static_cast<int64_t>(tensor.size(2)),
              static_cast<int64_t>(tensor.size(3)), static_cast<int64_t>(tensor.size(1))};
    else
      for (int i = 0; i < 4; ++i) dims[i] = tensor.size(i);
    INFINI_OPS_CNNL_CHECK(cnnlSetTensorDescriptor_v2(
        desc.get(), nhwc ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NCHW,
        cnnl_utils::GetDataType(tensor.dtype()), dims.size(), dims.data()));
    return desc;
  }

  static void SetupTranspose(Transpose& desc, const std::array<int, 4>& permutation) {
    cnnlTransposeDescriptor_t value = nullptr;
    INFINI_OPS_CNNL_CHECK(cnnlCreateTransposeDescriptor(&value));
    desc.reset(value);
    INFINI_OPS_CNNL_CHECK(cnnlSetTransposeDescriptor(value, permutation.size(), permutation.data()));
  }

  static cnrt_utils::DeviceBuffer Allocate(size_t bytes) {
    auto result = cnrt_utils::AllocateDeviceBuffer(bytes);
    if (bytes && !result)
      throw std::runtime_error("Cambricon Resize2d allocation failed");
    return result;
  }

  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_, input_nhwc_, out_, out_nhwc_;
  Transpose to_nhwc_{nullptr, cnnlDestroyTransposeDescriptor};
  Transpose to_nchw_{nullptr, cnnlDestroyTransposeDescriptor};
  std::unique_ptr<std::remove_pointer_t<cnnlInterpDescriptor_t>,
                  decltype(&cnnlDestroyInterpDescriptor)>
      interp_{nullptr, cnnlDestroyInterpDescriptor};
  cnrt_utils::DeviceBuffer input_buffer_, output_buffer_, scratch_;
  std::size_t input_workspace_ = 0, output_workspace_ = 0, scratch_size_ = 0;
  bool nearest2x_ = false;
};

}  // namespace infini::ops

#endif
