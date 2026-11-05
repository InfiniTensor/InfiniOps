#ifndef INFINI_OPS_CAMBRICON_INDEX_SELECT_KERNEL_H_
#define INFINI_OPS_CAMBRICON_INDEX_SELECT_KERNEL_H_

#include <stdexcept>

#include "base/gather_axis.h"
#include "base/index_select.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"
#include "native/cambricon/scatter_index.h"

namespace infini::ops {

template <class Base, bool WrapNegative>
class CambriconAxisSelection : public Base {
 public:
  CambriconAxisSelection(const Tensor input, const Tensor index, int64_t dim,
                         Tensor out)
      : Base{input, index, dim, out}, axis_(dim) {
    const auto rank = static_cast<int64_t>(input.ndim());
    if (axis_ < 0) axis_ += rank;
    if (axis_ < 0 || axis_ >= rank || index.ndim() != 1 || input.ndim() != out.ndim() ||
        (index.dtype() != DataType::kInt32 && index.dtype() != DataType::kInt64) ||
        input.dtype() != out.dtype() || input.device() != out.device() ||
        index.device() != out.device() || !input.IsContiguous() ||
        !index.IsContiguous() || !out.IsContiguous())
      throw std::invalid_argument("Cambricon IndexSelect: invalid tensor metadata");
    for (size_t i = 0; i < out.ndim(); ++i) {
      const auto expected = i == static_cast<size_t>(axis_) ? index.numel() : input.shape()[i];
      if (out.shape()[i] != expected)
        throw std::invalid_argument("Cambricon IndexSelect: invalid output shape");
    }
    if (out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    input_ = cnnl_utils::MakeTensorDescriptor(input);
    index_ = cnnl_utils::MakeTensorDescriptor(index);
    output_ = cnnl_utils::MakeTensorDescriptor(out);
    if constexpr (WrapNegative) {
      normalized_ = cnrt_utils::AllocateDeviceBuffer(index.numel() * index.element_size());
      if (!normalized_)
        throw std::runtime_error("Cambricon GatherAxis index allocation failed");
    }
  }

  void operator()(const Tensor input, const Tensor index, int64_t,
                  Tensor out) const override {
    if (!output_) return;
    auto queue = static_cast<cnrtQueue_t>(this->stream_);
    const void* indices = index.data();
    if constexpr (WrapNegative) {
      NormalizeAxisIndices(indices, normalized_.get(), input.size(axis_),
                           index.numel(), index.dtype(), queue);
      indices = normalized_.get();
    }
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), queue));
    INFINI_OPS_CNNL_CHECK(cnnlIndexSelect(
        handle_.get(), static_cast<int>(axis_), input_.get(), input.data(),
        index_.get(), indices, output_.get(), out.data()));
  }

 private:

  int64_t axis_;
  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_, index_, output_;
  cnrt_utils::DeviceBuffer normalized_;
};

template <>
class Operator<IndexSelect, Device::Type::kCambricon>
    : public CambriconAxisSelection<IndexSelect, false> {
  using CambriconAxisSelection::CambriconAxisSelection;
};

template <>
class Operator<GatherAxis, Device::Type::kCambricon>
    : public CambriconAxisSelection<GatherAxis, true> {
  using CambriconAxisSelection::CambriconAxisSelection;
};

}  // namespace infini::ops

#endif
