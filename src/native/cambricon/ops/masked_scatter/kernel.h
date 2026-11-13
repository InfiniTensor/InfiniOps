#ifndef INFINI_OPS_CAMBRICON_MASKED_SCATTER_KERNEL_H_
#define INFINI_OPS_CAMBRICON_MASKED_SCATTER_KERNEL_H_

#include "base/masked_scatter.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops {

template <>
class Operator<MaskedScatter, Device::Type::kCambricon> : public MaskedScatter {
 public:
  Operator(const Tensor input, const Tensor mask, const Tensor source, Tensor out)
      : MaskedScatter{input, mask, source, out} {
    if (input.shape() != out.shape() || input.dtype() != out.dtype() ||
        source.dtype() != out.dtype() || mask.dtype() != DataType::kBool ||
        input.device() != out.device() || mask.device() != out.device() ||
        source.device() != out.device() ||
        !input.IsContiguous() || !mask.IsContiguous() ||
        !source.IsContiguous() || !out.IsContiguous() || mask.ndim() > input.ndim())
      throw std::invalid_argument("Unsupported Cambricon MaskedScatter tensors");
    for (size_t i = 0; i < mask.ndim(); ++i) {
      const auto size = input.size(input.ndim() - mask.ndim() + i);
      if (mask.size(i) != 1 && mask.size(i) != size)
        throw std::invalid_argument("Cambricon MaskedScatter mask is not broadcastable");
    }
    if (out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    const auto input_shape = input.shape();
    const auto mask_shape = mask.shape();
    input_ = Describe(input.dtype(), input_shape);
    out_ = Describe(out.dtype(), out.shape());
    mask_ = Describe(mask.dtype(), mask_shape);
    source_ = Describe(source.dtype(), Tensor::Shape{source.numel()});
    if (mask_shape != input_shape) {
      expanded_mask_ = Describe(mask.dtype(), input_shape);
      mask_bytes_ = (input.numel() + 255) & ~size_t{255};
    }
    INFINI_OPS_CNNL_CHECK(cnnlGetMaskedWorkspaceSize(
        handle_.get(), CNNL_MASKED_SCATTER, input_.get(), EffectiveMask(),
        source_.get(), out_.get(), &masked_bytes_));
    scratch_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_in_bytes());
    if (workspace_size_in_bytes() && !scratch_)
      throw std::runtime_error("Cambricon MaskedScatter workspace allocation failed");
  }

  std::size_t workspace_size_in_bytes() const override { return mask_bytes_ + masked_bytes_; }

  void operator()(const Tensor input, const Tensor mask, const Tensor source,
                  Tensor out) const override {
    if (!out_) return;
    auto* scratch = static_cast<uint8_t*>(
        workspace_ && workspace_size_in_bytes_ >= workspace_size_in_bytes()
            ? workspace_ : scratch_.get());
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
    const void* mask_data = mask.data();
    if (expanded_mask_) {
      INFINI_OPS_CNNL_CHECK(cnnlExpand(handle_.get(), mask_.get(), mask_data,
                                      expanded_mask_.get(), scratch));
      mask_data = scratch;
    }
    INFINI_OPS_CNNL_CHECK(cnnlMasked_v2(
        handle_.get(), CNNL_MASKED_SCATTER, input_.get(), input.data(),
        EffectiveMask(), mask_data, source_.get(), source.data(),
        masked_bytes_ ? scratch + mask_bytes_ : nullptr, masked_bytes_,
        out_.get(), out.data()));
  }

 private:
  template <typename Shape>
  static cnnl_utils::TensorDescriptor Describe(DataType dtype, const Shape& shape) {
    auto desc = cnnl_utils::CreateTensorDescriptor();
    std::vector<int64_t> dims(shape.begin(), shape.end());
    if (dims.empty()) dims.push_back(1);
    INFINI_OPS_CNNL_CHECK(cnnlSetTensorDescriptor_v2(
        desc.get(), CNNL_LAYOUT_ARRAY, cnnl_utils::GetDataType(dtype),
        static_cast<int>(dims.size()), dims.data()));
    return desc;
  }

  cnnlTensorDescriptor_t EffectiveMask() const {
    return expanded_mask_ ? expanded_mask_.get() : mask_.get();
  }

  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_, mask_, source_, out_, expanded_mask_;
  cnrt_utils::DeviceBuffer scratch_;
  std::size_t mask_bytes_ = 0, masked_bytes_ = 0;
};

}  // namespace infini::ops

#endif
