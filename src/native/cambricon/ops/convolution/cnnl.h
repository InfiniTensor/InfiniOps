#ifndef INFINI_OPS_CAMBRICON_CONVOLUTION_CNNL_H_
#define INFINI_OPS_CAMBRICON_CONVOLUTION_CNNL_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "base/convolution.h"
#include "data_type.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {
namespace cambricon_convolution_detail {

constexpr std::size_t kWorkspaceAlignment = 128;

inline std::size_t AlignWorkspace(std::size_t size) {
  return (size + kWorkspaceAlignment - 1) / kWorkspaceAlignment *
         kWorkspaceAlignment;
}

struct ConvolutionDescriptorDeleter {
  using pointer = cnnlConvolutionDescriptor_t;

  void operator()(pointer desc) const noexcept {
    if (desc) {
      (void)cnnlDestroyConvolutionDescriptor(desc);
    }
  }
};

using ConvolutionDescriptor =
    std::unique_ptr<std::remove_pointer_t<cnnlConvolutionDescriptor_t>,
                    ConvolutionDescriptorDeleter>;

inline ConvolutionDescriptor CreateConvolutionDescriptor() {
  cnnlConvolutionDescriptor_t desc{nullptr};
  INFINI_OPS_CNNL_CHECK(cnnlCreateConvolutionDescriptor(&desc));
  return ConvolutionDescriptor{desc};
}

struct TransposeDescriptorDeleter {
  using pointer = cnnlTransposeDescriptor_t;

  void operator()(pointer desc) const noexcept {
    if (desc) {
      (void)cnnlDestroyTransposeDescriptor(desc);
    }
  }
};

using TransposeDescriptor =
    std::unique_ptr<std::remove_pointer_t<cnnlTransposeDescriptor_t>,
                    TransposeDescriptorDeleter>;

inline TransposeDescriptor MakeTransposeDescriptor(
    const std::vector<int>& permutation) {
  cnnlTransposeDescriptor_t desc{nullptr};
  INFINI_OPS_CNNL_CHECK(cnnlCreateTransposeDescriptor(&desc));
  TransposeDescriptor result{desc};
  INFINI_OPS_CNNL_CHECK(cnnlSetTransposeDescriptor(
      result.get(), static_cast<int>(permutation.size()), permutation.data()));
  return result;
}

struct QuantizeDescriptorDeleter {
  using pointer = cnnlQuantizeExDescriptor_t;

  void operator()(pointer desc) const noexcept {
    if (desc) {
      (void)cnnlDestroyQuantizeExDescriptor(desc);
    }
  }
};

using QuantizeDescriptor =
    std::unique_ptr<std::remove_pointer_t<cnnlQuantizeExDescriptor_t>,
                    QuantizeDescriptorDeleter>;

inline void SetTensorDescriptor(
    cnnlTensorDescriptor_t desc, cnnlTensorLayout_t layout,
    cnnlDataType_t dtype, const std::vector<std::int64_t>& shape,
    const std::vector<std::int64_t>* strides = nullptr) {
  if (strides) {
    assert(shape.size() == strides->size());
    INFINI_OPS_CNNL_CHECK(cnnlSetTensorDescriptorEx_v2(
        desc, layout, dtype, static_cast<int>(shape.size()), shape.data(),
        strides->data()));
  } else {
    INFINI_OPS_CNNL_CHECK(cnnlSetTensorDescriptor_v2(
        desc, layout, dtype, static_cast<int>(shape.size()), shape.data()));
  }
}

struct Shapes {
  std::vector<std::int64_t> input_original;
  std::vector<std::int64_t> input_cnnl;
  std::vector<std::int64_t> weight_original;
  std::vector<std::int64_t> weight_cnnl;
  std::vector<std::int64_t> out_cnnl;
  std::vector<int> padding;
  std::vector<int> stride;
  std::vector<int> dilation;
  std::vector<int> to_cnnl;
  cnnlTensorLayout_t cnnl_layout{CNNL_LAYOUT_ARRAY};
  int convolution_ndim{0};
};

inline std::vector<std::int64_t> ToInt64(const Tensor::Shape& values) {
  return {values.begin(), values.end()};
}

inline Shapes MakeShapes(const conv_detail::Metadata& metadata) {
  Shapes shapes;
  const auto& input = metadata.input_shape;
  const auto& weight = metadata.weight_shape;
  const auto& out = metadata.out_shape;

  if (metadata.spatial_ndim == 1) {
    shapes.input_original = {static_cast<std::int64_t>(input[0]),
                             static_cast<std::int64_t>(input[1]), 1,
                             static_cast<std::int64_t>(input[2])};
    shapes.input_cnnl = {static_cast<std::int64_t>(input[0]), 1,
                         static_cast<std::int64_t>(input[2]),
                         static_cast<std::int64_t>(input[1])};

    shapes.weight_original = {static_cast<std::int64_t>(weight[0]),
                              static_cast<std::int64_t>(weight[1]), 1,
                              static_cast<std::int64_t>(weight[2])};
    shapes.weight_cnnl = {static_cast<std::int64_t>(weight[0]), 1,
                          static_cast<std::int64_t>(weight[2]),
                          static_cast<std::int64_t>(weight[1])};

    shapes.out_cnnl = {static_cast<std::int64_t>(out[0]), 1,
                       static_cast<std::int64_t>(out[2]),
                       static_cast<std::int64_t>(out[1])};

    shapes.padding = {0, 0, static_cast<int>(metadata.padding[0]),
                      static_cast<int>(metadata.padding[0])};
    shapes.stride = {1, static_cast<int>(metadata.stride[0])};
    shapes.dilation = {1, static_cast<int>(metadata.dilation[0])};
    shapes.to_cnnl = {0, 2, 3, 1};
    shapes.cnnl_layout = CNNL_LAYOUT_NHWC;
    shapes.convolution_ndim = 4;
  } else if (metadata.spatial_ndim == 2) {
    shapes.input_original = ToInt64(metadata.input_shape);
    shapes.input_cnnl = {static_cast<std::int64_t>(input[0]),
                         static_cast<std::int64_t>(input[2]),
                         static_cast<std::int64_t>(input[3]),
                         static_cast<std::int64_t>(input[1])};

    shapes.weight_original = ToInt64(metadata.weight_shape);
    shapes.weight_cnnl = {static_cast<std::int64_t>(weight[0]),
                          static_cast<std::int64_t>(weight[2]),
                          static_cast<std::int64_t>(weight[3]),
                          static_cast<std::int64_t>(weight[1])};

    shapes.out_cnnl = {
        static_cast<std::int64_t>(out[0]), static_cast<std::int64_t>(out[2]),
        static_cast<std::int64_t>(out[3]), static_cast<std::int64_t>(out[1])};

    shapes.padding = {static_cast<int>(metadata.padding[0]),
                      static_cast<int>(metadata.padding[0]),
                      static_cast<int>(metadata.padding[1]),
                      static_cast<int>(metadata.padding[1])};
    shapes.stride = {static_cast<int>(metadata.stride[0]),
                     static_cast<int>(metadata.stride[1])};
    shapes.dilation = {static_cast<int>(metadata.dilation[0]),
                       static_cast<int>(metadata.dilation[1])};
    shapes.to_cnnl = {0, 2, 3, 1};
    shapes.cnnl_layout = CNNL_LAYOUT_NHWC;
    shapes.convolution_ndim = 4;
  } else {
    shapes.input_original = ToInt64(metadata.input_shape);
    shapes.input_cnnl = {static_cast<std::int64_t>(input[0]),
                         static_cast<std::int64_t>(input[2]),
                         static_cast<std::int64_t>(input[3]),
                         static_cast<std::int64_t>(input[4]),
                         static_cast<std::int64_t>(input[1])};

    shapes.weight_original = ToInt64(metadata.weight_shape);
    shapes.weight_cnnl = {static_cast<std::int64_t>(weight[0]),
                          static_cast<std::int64_t>(weight[2]),
                          static_cast<std::int64_t>(weight[3]),
                          static_cast<std::int64_t>(weight[4]),
                          static_cast<std::int64_t>(weight[1])};

    shapes.out_cnnl = {
        static_cast<std::int64_t>(out[0]), static_cast<std::int64_t>(out[2]),
        static_cast<std::int64_t>(out[3]), static_cast<std::int64_t>(out[4]),
        static_cast<std::int64_t>(out[1])};

    shapes.padding = {static_cast<int>(metadata.padding[0]),
                      static_cast<int>(metadata.padding[0]),
                      static_cast<int>(metadata.padding[1]),
                      static_cast<int>(metadata.padding[1]),
                      static_cast<int>(metadata.padding[2]),
                      static_cast<int>(metadata.padding[2])};
    shapes.stride = {static_cast<int>(metadata.stride[0]),
                     static_cast<int>(metadata.stride[1]),
                     static_cast<int>(metadata.stride[2])};
    shapes.dilation = {static_cast<int>(metadata.dilation[0]),
                       static_cast<int>(metadata.dilation[1]),
                       static_cast<int>(metadata.dilation[2])};
    shapes.to_cnnl = {0, 2, 3, 4, 1};
    shapes.cnnl_layout = CNNL_LAYOUT_NDHWC;
    shapes.convolution_ndim = 5;
  }

  return shapes;
}

}  // namespace cambricon_convolution_detail

template <>
class Operator<Convolution, Device::Type::kCambricon> : public Convolution {
 public:
  Operator(const Tensor input, const Tensor weight, std::optional<Tensor> bias,
           const std::vector<int64_t> stride,
           const std::vector<int64_t> padding,
           const std::vector<int64_t> dilation, const bool transposed,
           const std::vector<int64_t> output_padding, const int64_t groups,
           Tensor out)
      : Convolution{input,    weight,     bias,           stride, padding,
                    dilation, transposed, output_padding, groups, out} {
    Initialize(input, weight, bias, out);
  }

  void operator()(const Tensor input, const Tensor weight,
                  std::optional<Tensor> bias,
                  const std::vector<int64_t> /*stride*/,
                  const std::vector<int64_t> /*padding*/,
                  const std::vector<int64_t> /*dilation*/,
                  const bool /*transposed*/,
                  const std::vector<int64_t> /*output_padding*/,
                  const int64_t /*groups*/, Tensor out) const override {
    if (metadata_.output_size == 0) {
      return;
    }

    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(
        cnnl_handle_.get(), static_cast<cnrtQueue_t>(stream_ ? stream_ : 0)));

    void* workspace = workspace_ ? workspace_ : default_workspace_.get();
    const auto available_workspace =
        workspace_ ? workspace_size_in_bytes_ : workspace_size_;
    assert(available_workspace >= workspace_size_ &&
           "`CambriconConvolution` requires a sufficiently large workspace.");

    auto* bytes = static_cast<char*>(workspace);
    void* input_cnnl = bytes;
    void* weight_cnnl = bytes + input_cnnl_bytes_;
    void* out_cnnl = bytes + input_cnnl_bytes_ + weight_cnnl_bytes_;
    void* transpose_workspace =
        bytes + input_cnnl_bytes_ + weight_cnnl_bytes_ + out_cnnl_bytes_;
    void* convolution_workspace =
        static_cast<char*>(transpose_workspace) + transpose_workspace_size_;

    INFINI_OPS_CNNL_CHECK(cnnlTranspose_v2(
        cnnl_handle_.get(), input_transpose_desc_.get(),
        input_original_desc_.get(), input.data(), input_transposed_desc_.get(),
        input_cnnl, transpose_workspace, transpose_workspace_size_));
    INFINI_OPS_CNNL_CHECK(
        cnnlTranspose_v2(cnnl_handle_.get(), weight_transpose_desc_.get(),
                         weight_original_desc_.get(), weight.data(),
                         weight_transposed_desc_.get(), weight_cnnl,
                         transpose_workspace, transpose_workspace_size_));
    INFINI_OPS_CNNL_CHECK(cnnlConvolutionForward(
        cnnl_handle_.get(), convolution_desc_.get(), algorithm_, nullptr,
        input_cnnl_desc_.get(), input_cnnl, weight_cnnl_desc_.get(),
        weight_cnnl, bias_desc_.get(),
        bias.has_value() ? bias->data() : nullptr, convolution_workspace,
        convolution_workspace_size_, nullptr, out_cnnl_desc_.get(), out_cnnl));
    CNRT_CHECK(cnrtMemcpyAsync(
        out.data(), out_cnnl, out.numel() * out.element_size(),
        static_cast<cnrtQueue_t>(stream_ ? stream_ : 0), cnrtMemcpyDevToDev));
  }

  std::size_t workspace_size_in_bytes() const override {
    return workspace_size_;
  }

 private:
  void Initialize(const Tensor input, const Tensor weight,
                  std::optional<Tensor> bias, const Tensor out) {
    if (metadata_.output_size == 0) {
      return;
    }

    assert(input.IsContiguous() && weight.IsContiguous() &&
           out.IsContiguous() && (!bias.has_value() || bias->IsContiguous()) &&
           "`CambriconConvolution` currently requires contiguous tensors.");

    const auto shapes = cambricon_convolution_detail::MakeShapes(metadata_);
    const auto dtype = cnnl_utils::GetDataType(metadata_.out_type);
    assert(dtype != CNNL_DTYPE_INVALID);

    cnnl_handle_ = cnnl_utils::CreateHandle();
    input_original_desc_ = cnnl_utils::CreateTensorDescriptor();
    input_transposed_desc_ = cnnl_utils::CreateTensorDescriptor();
    input_cnnl_desc_ = cnnl_utils::CreateTensorDescriptor();
    weight_original_desc_ = cnnl_utils::CreateTensorDescriptor();
    weight_transposed_desc_ = cnnl_utils::CreateTensorDescriptor();
    weight_cnnl_desc_ = cnnl_utils::CreateTensorDescriptor();
    out_cnnl_desc_ = cnnl_utils::CreateTensorDescriptor();

    cambricon_convolution_detail::SetTensorDescriptor(
        input_original_desc_.get(), CNNL_LAYOUT_ARRAY, dtype,
        shapes.input_original);
    cambricon_convolution_detail::SetTensorDescriptor(
        input_transposed_desc_.get(), CNNL_LAYOUT_ARRAY, dtype,
        shapes.input_cnnl);
    cambricon_convolution_detail::SetTensorDescriptor(
        input_cnnl_desc_.get(), shapes.cnnl_layout, dtype, shapes.input_cnnl);
    cambricon_convolution_detail::SetTensorDescriptor(
        weight_original_desc_.get(), CNNL_LAYOUT_ARRAY, dtype,
        shapes.weight_original);
    cambricon_convolution_detail::SetTensorDescriptor(
        weight_transposed_desc_.get(), CNNL_LAYOUT_ARRAY, dtype,
        shapes.weight_cnnl);
    cambricon_convolution_detail::SetTensorDescriptor(
        weight_cnnl_desc_.get(), shapes.cnnl_layout, dtype, shapes.weight_cnnl);
    cambricon_convolution_detail::SetTensorDescriptor(
        out_cnnl_desc_.get(), shapes.cnnl_layout, dtype, shapes.out_cnnl);

    if (bias.has_value()) {
      bias_desc_ = cnnl_utils::CreateTensorDescriptor();
      const std::vector<std::int64_t> bias_shape{
          static_cast<std::int64_t>(metadata_.bias_shape[0])};
      cambricon_convolution_detail::SetTensorDescriptor(
          bias_desc_.get(), CNNL_LAYOUT_ARRAY, dtype, bias_shape);
    }

    convolution_desc_ =
        cambricon_convolution_detail::CreateConvolutionDescriptor();
    const auto compute_dtype =
        metadata_.out_type == DataType::kBFloat16 ? CNNL_DTYPE_FLOAT : dtype;
    INFINI_OPS_CNNL_CHECK(cnnlSetConvolutionDescriptor(
        convolution_desc_.get(), shapes.convolution_ndim, shapes.padding.data(),
        shapes.stride.data(), shapes.dilation.data(),
        static_cast<int>(metadata_.groups), compute_dtype));

    if (metadata_.out_type == DataType::kBFloat16) {
      cnnlQuantizeExDescriptor_t quantize_desc{nullptr};
      INFINI_OPS_CNNL_CHECK(cnnlCreateQuantizeExDescriptor(&quantize_desc));
      out_quantize_desc_.reset(quantize_desc);
      INFINI_OPS_CNNL_CHECK(cnnlSetQuantizeExDescriptorQuantSchemeAndDtype(
          out_quantize_desc_.get(), CNNL_QUANTIZE_NONE, CNNL_DTYPE_FLOAT));
      INFINI_OPS_CNNL_CHECK(cnnlSetConvolutionDescriptorQuant(
          convolution_desc_.get(), nullptr, nullptr, out_quantize_desc_.get()));
    }

    input_transpose_desc_ =
        cambricon_convolution_detail::MakeTransposeDescriptor(shapes.to_cnnl);
    weight_transpose_desc_ =
        cambricon_convolution_detail::MakeTransposeDescriptor(shapes.to_cnnl);

    std::size_t input_transpose_workspace = 0;
    std::size_t weight_transpose_workspace = 0;
    INFINI_OPS_CNNL_CHECK(cnnlGetTransposeWorkspaceSize(
        cnnl_handle_.get(), input_original_desc_.get(),
        input_transpose_desc_.get(), &input_transpose_workspace));
    INFINI_OPS_CNNL_CHECK(cnnlGetTransposeWorkspaceSize(
        cnnl_handle_.get(), weight_original_desc_.get(),
        weight_transpose_desc_.get(), &weight_transpose_workspace));
    INFINI_OPS_CNNL_CHECK(cnnlGetConvolutionForwardWorkspaceSize(
        cnnl_handle_.get(), input_cnnl_desc_.get(), weight_cnnl_desc_.get(),
        out_cnnl_desc_.get(), bias_desc_.get(), convolution_desc_.get(),
        algorithm_, &convolution_workspace_size_));

    input_cnnl_bytes_ = cambricon_convolution_detail::AlignWorkspace(
        input.numel() * input.element_size());
    weight_cnnl_bytes_ = cambricon_convolution_detail::AlignWorkspace(
        weight.numel() * weight.element_size());
    out_cnnl_bytes_ = cambricon_convolution_detail::AlignWorkspace(
        out.numel() * out.element_size());
    transpose_workspace_size_ = cambricon_convolution_detail::AlignWorkspace(
        std::max(input_transpose_workspace, weight_transpose_workspace));
    convolution_workspace_size_ = cambricon_convolution_detail::AlignWorkspace(
        convolution_workspace_size_);
    workspace_size_ = input_cnnl_bytes_ + weight_cnnl_bytes_ + out_cnnl_bytes_ +
                      transpose_workspace_size_ + convolution_workspace_size_;
    default_workspace_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
  }

  std::size_t input_cnnl_bytes_{0};

  std::size_t weight_cnnl_bytes_{0};

  std::size_t out_cnnl_bytes_{0};

  std::size_t transpose_workspace_size_{0};

  std::size_t convolution_workspace_size_{0};

  std::size_t workspace_size_{0};

  cnrt_utils::DeviceBuffer default_workspace_{};

  cnnl_utils::Handle cnnl_handle_{};

  cnnl_utils::TensorDescriptor input_original_desc_{};

  cnnl_utils::TensorDescriptor input_transposed_desc_{};

  cnnl_utils::TensorDescriptor input_cnnl_desc_{};

  cnnl_utils::TensorDescriptor weight_original_desc_{};

  cnnl_utils::TensorDescriptor weight_transposed_desc_{};

  cnnl_utils::TensorDescriptor weight_cnnl_desc_{};

  cnnl_utils::TensorDescriptor out_cnnl_desc_{};

  cnnl_utils::TensorDescriptor bias_desc_{};

  cambricon_convolution_detail::ConvolutionDescriptor convolution_desc_{};

  cambricon_convolution_detail::QuantizeDescriptor out_quantize_desc_{};

  cambricon_convolution_detail::TransposeDescriptor input_transpose_desc_{};

  cambricon_convolution_detail::TransposeDescriptor weight_transpose_desc_{};

  cnnlConvolutionForwardAlgo_t algorithm_{CNNL_CONVOLUTION_FWD_ALGO_DIRECT};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CAMBRICON_CONVOLUTION_CNNL_H_
