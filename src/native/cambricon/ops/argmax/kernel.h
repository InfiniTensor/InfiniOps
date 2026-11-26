#ifndef INFINI_OPS_CAMBRICON_ARGMAX_KERNEL_H_
#define INFINI_OPS_CAMBRICON_ARGMAX_KERNEL_H_
#include <climits>
#include <cstdint>
#include <stdexcept>

#include "base/argmax.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/common.h"
#include "native/cambricon/cnrt_utils.h"
namespace infini::ops {
void CambriconArgmaxLaunch(cnrtQueue_t, const void*, void*, int32_t,
                           std::size_t, uint32_t);
template <>
class Operator<Argmax, Device::Type::kCambricon> : public Argmax {
 public:
  static bool supports(const Tensor x, const std::optional<int64_t> dim,
                       bool keepdim, const Tensor y) {
    if (dim) {
      const auto rank = static_cast<int64_t>(x.ndim());
      if (x.device().type() != Device::Type::kCambricon ||
          x.device() != y.device() || !x.IsContiguous() || !y.IsContiguous() ||
          (x.dtype() != DataType::kFloat32 && x.dtype() != DataType::kFloat16 &&
           x.dtype() != DataType::kInt32) ||
          y.dtype() != DataType::kInt64 || *dim < -rank || *dim >= rank)
        return false;
      const auto axis = static_cast<size_t>(*dim < 0 ? *dim + rank : *dim);
      if (x.shape()[axis] == 0) return false;
      Tensor::Shape expected;
      for (size_t i = 0; i < x.ndim(); ++i) {
        if (i != axis || keepdim) expected.push_back(i == axis ? 1 : x.shape()[i]);
      }
      return expected == y.shape();
    }
    if (x.device().type() != Device::Type::kCambricon ||
        x.device().index() < 0 || x.dtype() != DataType::kFloat32 ||
        !x.numel() || x.numel() > INT_MAX || !x.IsContiguous() || !x.data() ||
        keepdim || x.device() != y.device() || y.ndim() != 0 ||
        y.dtype() != DataType::kInt64 || !y.IsContiguous() || !y.data())
      return false;
    const auto a = reinterpret_cast<uintptr_t>(x.data()),
               b = reinterpret_cast<uintptr_t>(y.data());
    if (a % sizeof(float) || b % alignof(int64_t)) return false;
    return b > a ? b - a >= x.numel() * sizeof(float)
                 : a - b >= sizeof(int64_t);
  }
  Operator(const Tensor x, const std::optional<int64_t> dim, bool keepdim,
           Tensor y)
      : Argmax(x, dim, keepdim, y) {
    if (!dim) return;
    if (!supports(x, dim, keepdim, y))
      throw std::invalid_argument("Unsupported Cambricon axis Argmax");
    if (!y.numel()) return;
    const auto rank = static_cast<int64_t>(x.ndim());
    axis_ = static_cast<int>(*dim < 0 ? *dim + rank : *dim);
    auto top_shape = input_shape_;
    top_shape[axis_] = 1;
    Tensor::Strides top_strides(top_shape.size());
    Tensor::Stride stride = 1;
    for (size_t i = top_shape.size(); i-- > 0;) {
      top_strides[i] = stride;
      stride *= top_shape[i];
    }
    handle_ = cnnl_utils::CreateHandle();
    input_ = cnnl_utils::MakeTensorDescriptor(input_type_, input_shape_, input_strides_);
    values_ = cnnl_utils::MakeTensorDescriptor(input_type_, top_shape, top_strides);
    indices_ = cnnl_utils::MakeTensorDescriptor(out_type_, top_shape, top_strides);
    INFINI_OPS_CNNL_CHECK(cnnlGetTopKTensorWorkspaceSize(
        handle_.get(), input_.get(), 1, axis_, true, values_.get(),
        indices_.get(), &workspace_size_));
    workspace_own_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
    values_storage_ = cnrt_utils::AllocateDeviceBuffer(y.numel() * x.element_size());
    if ((workspace_size_ && !workspace_own_) || !values_storage_)
      throw std::runtime_error("Cambricon Argmax workspace allocation failed");
  }
  std::size_t workspace_size_in_bytes() const override { return workspace_size_; }
  void operator()(const Tensor x, const std::optional<int64_t> dim,
                  bool keepdim, Tensor y) const override {
    if (!supports(x, dim, keepdim, y) || x.shape() != input_shape_ ||
        x.strides() != input_strides_ || x.dtype() != input_type_ ||
        y.shape() != out_shape_ || y.strides() != out_strides_ ||
        y.dtype() != out_type_ || x.device().index() != device_index_ ||
        dim != dim_ || keepdim != keepdim_)
      throw std::invalid_argument(
          "Cambricon Argmax prepared metadata changed");
    if (dim) {
      if (!indices_) return;
      INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), static_cast<cnrtQueue_t>(stream_)));
      auto* scratch = workspace_ && workspace_size_in_bytes_ >= workspace_size_
                          ? workspace_ : workspace_own_.get();
      INFINI_OPS_CNNL_CHECK(cnnlTopKTensor_v3(
          handle_.get(), input_.get(), x.data(), 1, axis_, true, true, true,
          scratch, workspace_size_, values_.get(), values_storage_.get(),
          indices_.get(), y.data()));
      return;
    }
    CambriconArgmaxLaunch(static_cast<cnrtQueue_t>(stream_), x.data(), y.data(),
                          static_cast<int32_t>(x.numel()), sizeof(float),
                          0x7f800000U);
    CNRT_CHECK(cnrtGetLastError());
  }

 private:
  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_, values_, indices_;
  cnrt_utils::DeviceBuffer workspace_own_, values_storage_;
  std::size_t workspace_size_ = 0;
  int axis_ = 0;
};
}  // namespace infini::ops
#endif
