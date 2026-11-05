#ifndef INFINI_OPS_CAMBRICON_INSTANCE_NORM_KERNEL_H_
#define INFINI_OPS_CAMBRICON_INSTANCE_NORM_KERNEL_H_

#include <algorithm>
#include <cmath>

#include "base/instance_norm.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<InstanceNorm, Device::Type::kCambricon> : public InstanceNorm {
 public:
  Operator(const Tensor input, const Tensor scale, const Tensor bias,
           float epsilon, Tensor out)
      : InstanceNorm{input, scale, bias, epsilon, out} {
    const auto rank = input.ndim();
    if ((rank != 3 && rank != 4) || input.shape() != out.shape() ||
        input.dtype() != out.dtype() ||
        (out.dtype() != DataType::kFloat32 && out.dtype() != DataType::kFloat16) ||
        scale.ndim() != 1 || bias.ndim() != 1 || scale.size(0) != input.size(1) ||
        bias.size(0) != input.size(1) || scale.dtype() != input.dtype() ||
        bias.dtype() != input.dtype() || input.device() != out.device() ||
        scale.device() != out.device() || bias.device() != out.device() ||
        !input.IsContiguous() || !out.IsContiguous() ||
        !scale.IsContiguous() || !bias.IsContiguous() || !std::isfinite(epsilon) || epsilon < 0)
      throw std::invalid_argument("Unsupported Cambricon InstanceNorm parameters");
    if (out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    const auto metadata = input.shape();
    const std::vector<int64_t> shape(metadata.begin(), metadata.end());
    std::vector<int> forward = rank == 3 ? std::vector<int>{0, 2, 1} : std::vector<int>{0, 2, 3, 1};
    std::vector<int> backward = rank == 3 ? forward : std::vector<int>{0, 3, 1, 2};
    std::vector<int64_t> permuted;
    for (auto dim : forward) permuted.push_back(shape[dim]);
    const auto dtype = cnnl_utils::GetDataType(input.dtype());
    first_ = Describe(CNNL_LAYOUT_ARRAY, dtype, shape);
    last_ = Describe(rank == 3 ? CNNL_LAYOUT_NLC : CNNL_LAYOUT_NHWC, dtype, permuted);
    channel_ = Describe(CNNL_LAYOUT_ARRAY, dtype, {shape[1]});
    statistics_ = Describe(CNNL_LAYOUT_NC, dtype, {shape[0], shape[1]});
    SetupTranspose(to_last_, forward);
    SetupTranspose(to_first_, backward);
    INFINI_OPS_CNNL_CHECK(cnnlGetTransposeWorkspaceSize(
        handle_.get(), first_.get(), to_last_.get(), &forward_bytes_));
    INFINI_OPS_CNNL_CHECK(cnnlGetTransposeWorkspaceSize(
        handle_.get(), last_.get(), to_first_.get(), &backward_bytes_));
    INFINI_OPS_CNNL_CHECK(cnnlGetInstanceNormForwardWorkspaceSize(
        handle_.get(), last_.get(), &normalization_bytes_));
    scratch_bytes_ = std::max({forward_bytes_, backward_bytes_, normalization_bytes_});
    input_last_ = Allocate(input.numel() * input.element_size());
    output_last_ = Allocate(out.numel() * out.element_size());
    const auto stats_bytes = static_cast<size_t>(shape[0]) * shape[1] * sizeof(float);
    saved_mean_ = Allocate(stats_bytes);
    saved_inverse_ = Allocate(stats_bytes);
    scratch_ = Allocate(scratch_bytes_);
  }

  std::size_t workspace_size_in_bytes() const override { return scratch_bytes_; }

  void operator()(const Tensor input, const Tensor scale, const Tensor bias,
                  float epsilon, Tensor out) const override {
    if (!first_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    auto* scratch = workspace_ && workspace_size_in_bytes_ >= scratch_bytes_
                        ? workspace_ : scratch_.get();
    INFINI_OPS_CNNL_CHECK(cnnlTranspose_v2(
        handle_.get(), to_last_.get(), first_.get(), input.data(), last_.get(),
        input_last_.get(), scratch, forward_bytes_));
    INFINI_OPS_CNNL_CHECK(cnnlInstanceNormForward(
        handle_.get(), epsilon, 0.0f, channel_.get(), scale.data(), last_.get(),
        input_last_.get(), bias.data(), statistics_.get(), nullptr, nullptr,
        scratch, normalization_bytes_, last_.get(), output_last_.get(),
        saved_mean_.get(), saved_inverse_.get()));
    INFINI_OPS_CNNL_CHECK(cnnlTranspose_v2(
        handle_.get(), to_first_.get(), last_.get(), output_last_.get(),
        first_.get(), out.data(), scratch, backward_bytes_));
  }

 private:
  using Transpose = std::unique_ptr<std::remove_pointer_t<cnnlTransposeDescriptor_t>,
                                    decltype(&cnnlDestroyTransposeDescriptor)>;
  static cnnl_utils::TensorDescriptor Describe(cnnlTensorLayout_t layout,
      cnnlDataType_t dtype, const std::vector<int64_t>& shape) {
    auto desc = cnnl_utils::CreateTensorDescriptor();
    INFINI_OPS_CNNL_CHECK(cnnlSetTensorDescriptor_v2(desc.get(), layout, dtype,
                                                    shape.size(), shape.data()));
    return desc;
  }

  static void SetupTranspose(Transpose& desc, const std::vector<int>& permutation) {
    cnnlTransposeDescriptor_t value = nullptr;
    INFINI_OPS_CNNL_CHECK(cnnlCreateTransposeDescriptor(&value));
    desc.reset(value);
    INFINI_OPS_CNNL_CHECK(cnnlSetTransposeDescriptor(value, permutation.size(), permutation.data()));
  }

  static cnrt_utils::DeviceBuffer Allocate(size_t bytes) {
    auto result = cnrt_utils::AllocateDeviceBuffer(bytes);
    if (bytes && !result)
      throw std::runtime_error("Cambricon InstanceNorm allocation failed");
    return result;
  }

  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor first_, last_, channel_, statistics_;
  Transpose to_last_{nullptr, cnnlDestroyTransposeDescriptor};
  Transpose to_first_{nullptr, cnnlDestroyTransposeDescriptor};
  cnrt_utils::DeviceBuffer input_last_, output_last_, saved_mean_, saved_inverse_, scratch_;
  std::size_t forward_bytes_ = 0, backward_bytes_ = 0, normalization_bytes_ = 0, scratch_bytes_ = 0;
};

}  // namespace infini::ops

#endif
