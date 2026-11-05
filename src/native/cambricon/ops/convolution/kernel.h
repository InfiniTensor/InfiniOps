#ifndef INFINI_OPS_CAMBRICON_CONVOLUTION_KERNEL_H_
#define INFINI_OPS_CAMBRICON_CONVOLUTION_KERNEL_H_

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "base/convolution.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<Convolution, Device::Type::kCambricon> : public Convolution {
 public:
  Operator(const Tensor input, const Tensor weight, std::optional<Tensor> bias,
           std::vector<int64_t> stride, std::vector<int64_t> padding,
           std::vector<int64_t> dilation, bool transposed,
           std::vector<int64_t> output_padding, int64_t groups, Tensor out)
      : Convolution{Validate(input, weight, bias, stride, padding, dilation,
                             transposed, output_padding, groups, out),
                    weight, bias, stride, padding, dilation, transposed,
                    output_padding, groups, out},
        transposed_(transposed) {
    if (out.numel() == 0) return;
    const auto rank = input.ndim();
    std::vector<int> forward{0}, backward{0, static_cast<int>(rank - 1)};
    for (size_t i = 2; i < rank; ++i) {
      forward.push_back(static_cast<int>(i));
      backward.push_back(static_cast<int>(i - 1));
    }
    forward.push_back(1);
    const auto channels_first = rank == 4 ? CNNL_LAYOUT_NCHW : CNNL_LAYOUT_NCDHW;
    const auto channels_last = rank == 4 ? CNNL_LAYOUT_NHWC : CNNL_LAYOUT_NDHWC;
    const auto dtype = cnnl_utils::GetDataType(input.dtype());
    handle_ = cnnl_utils::CreateHandle();
    if (!IsChannelsLast(input)) input_first_ = MakeDescriptor(input, dtype, channels_first);
    if (!IsChannelsLast(weight)) weight_first_ = MakeDescriptor(weight, dtype, channels_first);
    if (!IsChannelsLast(out)) output_first_ = MakeDescriptor(out, dtype, channels_first);
    input_last_ = MakeDescriptor(input, dtype, channels_last, forward);
    weight_last_ = MakeDescriptor(weight, dtype, channels_last, forward);
    output_last_ = MakeDescriptor(out, dtype, channels_last, forward);
    if (bias) {
      bias_.reset(CreateTensorDescriptor());
      std::vector<int64_t> shape(rank, 1);
      shape.back() = bias->numel();
      INFINI_OPS_CNNL_CHECK(cnnlSetTensorDescriptor_v2(
          bias_.get(), channels_last, dtype, rank, shape.data()));
    }

    MakeTranspose(to_last_, forward);
    MakeTranspose(to_first_, backward);
    cnnlConvolutionDescriptor_t conv = nullptr;
    INFINI_OPS_CNNL_CHECK(cnnlCreateConvolutionDescriptor(&conv));
    conv_.reset(conv);
    std::vector<int> pad, step, dilate;
    for (size_t i = 0; i < rank - 2; ++i) {
      pad.push_back(static_cast<int>(padding[i]));
      pad.push_back(static_cast<int>(padding[i]));
      step.push_back(static_cast<int>(stride[i]));
      dilate.push_back(static_cast<int>(dilation[i]));
    }
    INFINI_OPS_CNNL_CHECK(cnnlSetConvolutionDescriptor(
        conv_.get(), rank, pad.data(), step.data(), dilate.data(), groups,
        CNNL_DTYPE_FLOAT));
    if (transposed_) {
      INFINI_OPS_CNNL_CHECK(cnnlGetConvolutionBackwardDataAlgorithm(
          handle_.get(), input_last_.get(), weight_last_.get(), conv_.get(),
          output_last_.get(), CNNL_CONVOLUTION_BWD_DATA_FASTEST, &transpose_algorithm_));
      INFINI_OPS_CNNL_CHECK(cnnlGetConvolutionBackwardDataWorkspaceSize(
          handle_.get(), weight_last_.get(), input_last_.get(), conv_.get(),
          output_last_.get(), transpose_algorithm_, &conv_workspace_));
    } else {
      INFINI_OPS_CNNL_CHECK(cnnlGetConvolutionForwardAlgorithm(
          handle_.get(), conv_.get(), input_last_.get(), weight_last_.get(),
          output_last_.get(), CNNL_CONVOLUTION_FWD_FASTEST, &algorithm_));
      INFINI_OPS_CNNL_CHECK(cnnlGetConvolutionForwardWorkspaceSize(
          handle_.get(), input_last_.get(), weight_last_.get(), output_last_.get(),
          bias_.get(), conv_.get(), algorithm_, &conv_workspace_));
    }
    if (input_first_) input_workspace_ = TransposeWorkspace(input_first_.get(), to_last_.get());
    if (weight_first_) weight_workspace_ = TransposeWorkspace(weight_first_.get(), to_last_.get());
    if (output_first_) output_workspace_ = TransposeWorkspace(output_last_.get(), to_first_.get());
    scratch_size_ = std::max({conv_workspace_, input_workspace_, weight_workspace_, output_workspace_});
    const auto bytes = ::infini::rt::kDataTypeToSize.at(input.dtype());
    if (input_first_) data_last_ = Allocate(input.numel() * bytes);
    if (weight_first_) filter_last_ = Allocate(weight.numel() * bytes);
    if (output_first_) result_last_ = Allocate(out.numel() * bytes);
    scratch_ = Allocate(scratch_size_);
  }

  std::size_t workspace_size_in_bytes() const override { return scratch_size_; }

  void operator()(const Tensor input, const Tensor weight, std::optional<Tensor> bias,
                  const std::vector<int64_t>, const std::vector<int64_t>,
                  const std::vector<int64_t>, bool, const std::vector<int64_t>,
                  int64_t, Tensor out) const override {
    if (!output_last_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    auto* scratch = workspace_ && workspace_size_in_bytes_ >= scratch_size_
                        ? workspace_ : scratch_.get();
    if (input_first_) INFINI_OPS_CNNL_CHECK(cnnlTranspose_v2(
        handle_.get(), to_last_.get(), input_first_.get(), input.data(),
        input_last_.get(), data_last_.get(), scratch, input_workspace_));
    if (weight_first_) INFINI_OPS_CNNL_CHECK(cnnlTranspose_v2(
        handle_.get(), to_last_.get(), weight_first_.get(), weight.data(),
        weight_last_.get(), filter_last_.get(), scratch, weight_workspace_));
    const auto* input_data = input_first_ ? data_last_.get() : input.data();
    const auto* weight_data = weight_first_ ? filter_last_.get() : weight.data();
    auto* output_data = output_first_ ? result_last_.get() : out.data();
    if (transposed_) {
      INFINI_OPS_CNNL_CHECK(cnnlConvolutionBackwardData(
          handle_.get(), nullptr, weight_last_.get(), weight_data,
          input_last_.get(), input_data, conv_.get(), transpose_algorithm_,
          scratch, conv_workspace_, nullptr, output_last_.get(), output_data));
    } else {
      INFINI_OPS_CNNL_CHECK(cnnlConvolutionForward(
          handle_.get(), conv_.get(), algorithm_, nullptr, input_last_.get(),
          input_data, weight_last_.get(), weight_data, bias_.get(),
          bias ? bias->data() : nullptr, scratch, conv_workspace_, nullptr,
          output_last_.get(), output_data));
    }
    if (output_first_) INFINI_OPS_CNNL_CHECK(cnnlTranspose_v2(
        handle_.get(), to_first_.get(), output_last_.get(), result_last_.get(),
        output_first_.get(), out.data(), scratch, output_workspace_));
  }

 private:
  using Transpose = std::unique_ptr<std::remove_pointer_t<cnnlTransposeDescriptor_t>,
                                    decltype(&cnnlDestroyTransposeDescriptor)>;

  static bool IsChannelsLast(const Tensor& tensor) {
    const auto rank = tensor.ndim();
    if (rank != 4 && rank != 5) return false;
    int64_t stride = 1;
    for (size_t i = 0; i < rank; ++i) {
      const size_t axis = i == 0 ? 1 : i == rank - 1 ? 0 : rank - i;
      if (tensor.shape()[axis] > 1 && tensor.stride(axis) != stride) return false;
      stride *= tensor.shape()[axis];
    }
    return true;
  }

  static bool IsSupportedLayout(const Tensor& tensor) {
    return tensor.IsContiguous() || IsChannelsLast(tensor);
  }

  static Tensor Validate(const Tensor& input, const Tensor& weight,
                         const std::optional<Tensor>& bias,
                         const std::vector<int64_t>& stride,
                         const std::vector<int64_t>& padding,
                         const std::vector<int64_t>& dilation, bool transposed,
                         const std::vector<int64_t>& output_padding,
                         int64_t groups, const Tensor& out) {
    const auto rank = input.ndim();
    if ((rank != 4 && rank != 5) || weight.ndim() != rank || out.ndim() != rank ||
        stride.size() != rank - 2 || padding.size() != rank - 2 ||
        dilation.size() != rank - 2 || output_padding.size() != rank - 2 ||
        (transposed && (rank != 4 || bias)) ||
        groups < 1 || groups > std::numeric_limits<int>::max() ||
        (input.dtype() != DataType::kFloat32 && input.dtype() != DataType::kFloat16) ||
        input.dtype() != weight.dtype() || input.dtype() != out.dtype() ||
        input.device() != weight.device() || input.device() != out.device() ||
        !IsSupportedLayout(input) || !IsSupportedLayout(weight) || !IsSupportedLayout(out) ||
        (bias && (bias->ndim() != 1 || bias->numel() != weight.shape()[0] ||
                  bias->dtype() != out.dtype() || bias->device() != out.device() ||
                  !bias->IsContiguous())))
      throw std::invalid_argument("Unsupported Cambricon Convolution tensor metadata");
    for (size_t i = 0; i < rank - 2; ++i) {
      const auto limit = std::numeric_limits<int>::max();
      if (stride[i] <= 0 || stride[i] > limit || dilation[i] <= 0 ||
          dilation[i] > limit || padding[i] < 0 || padding[i] > limit ||
          output_padding[i] < 0 ||
          (transposed ? output_padding[i] >= stride[i] : output_padding[i] != 0))
        throw std::invalid_argument("Unsupported Cambricon Convolution spatial parameters");
    }
    const auto input_channels = input.shape()[1], output_channels = out.shape()[1];
    if (input_channels % groups || output_channels % groups ||
        (transposed ? (weight.shape()[0] != input_channels ||
                       weight.shape()[1] != output_channels / groups)
                    : (weight.shape()[0] != output_channels ||
                       weight.shape()[1] != input_channels / groups)) ||
        out.shape()[0] != input.shape()[0])
      throw std::invalid_argument("Invalid Cambricon Convolution channels");
    return input;
  }

  static cnnlTensorDescriptor_t CreateTensorDescriptor() {
    cnnlTensorDescriptor_t desc = nullptr;
    INFINI_OPS_CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));
    return desc;
  }

  static cnnl_utils::TensorDescriptor MakeDescriptor(
      const Tensor& tensor, cnnlDataType_t dtype, cnnlTensorLayout_t layout,
      const std::vector<int>& permutation = {}) {
    cnnl_utils::TensorDescriptor result{CreateTensorDescriptor()};
    std::vector<int64_t> dims(tensor.ndim());
    for (size_t i = 0; i < dims.size(); ++i)
      dims[i] = tensor.shape()[permutation.empty() ? i : permutation[i]];
    INFINI_OPS_CNNL_CHECK(cnnlSetTensorDescriptor_v2(
        result.get(), layout, dtype, dims.size(), dims.data()));
    return result;
  }

  static void MakeTranspose(Transpose& descriptor, const std::vector<int>& permutation) {
    cnnlTransposeDescriptor_t value = nullptr;
    INFINI_OPS_CNNL_CHECK(cnnlCreateTransposeDescriptor(&value));
    descriptor.reset(value);
    INFINI_OPS_CNNL_CHECK(cnnlSetTransposeDescriptor(
        value, permutation.size(), permutation.data()));
  }

  std::size_t TransposeWorkspace(cnnlTensorDescriptor_t input,
                                 cnnlTransposeDescriptor_t transpose) const {
    std::size_t bytes = 0;
    INFINI_OPS_CNNL_CHECK(cnnlGetTransposeWorkspaceSize(handle_.get(), input, transpose, &bytes));
    return bytes;
  }

  static cnrt_utils::DeviceBuffer Allocate(std::size_t bytes) {
    auto result = cnrt_utils::AllocateDeviceBuffer(bytes);
    if (bytes && !result) throw std::runtime_error("Cambricon Convolution allocation failed");
    return result;
  }

  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_first_, weight_first_, output_first_;
  cnnl_utils::TensorDescriptor input_last_, weight_last_, output_last_, bias_;
  Transpose to_last_{nullptr, cnnlDestroyTransposeDescriptor};
  Transpose to_first_{nullptr, cnnlDestroyTransposeDescriptor};
  std::unique_ptr<std::remove_pointer_t<cnnlConvolutionDescriptor_t>,
                  decltype(&cnnlDestroyConvolutionDescriptor)>
      conv_{nullptr, cnnlDestroyConvolutionDescriptor};
  cnnlConvolutionForwardAlgo_t algorithm_{};
  cnnlConvolutionBwdDataAlgo_t transpose_algorithm_{};
  bool transposed_;
  cnrt_utils::DeviceBuffer data_last_, filter_last_, result_last_, scratch_;
  std::size_t input_workspace_ = 0, weight_workspace_ = 0;
  std::size_t output_workspace_ = 0, conv_workspace_ = 0, scratch_size_ = 0;
};

}  // namespace infini::ops

#endif
