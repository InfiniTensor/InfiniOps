#ifndef INFINI_OPS_CAMBRICON_SCATTER_ND_KERNEL_H_
#define INFINI_OPS_CAMBRICON_SCATTER_ND_KERNEL_H_

#include "base/scatter_nd.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"
#include "native/cambricon/scatter_index.h"

namespace infini::ops {

template <>
class Operator<ScatterNd, Device::Type::kCambricon> : public ScatterNd {
 public:
  Operator(const Tensor input, const Tensor indices, const Tensor updates,
           bool add, Tensor out)
      : ScatterNd{input, indices, updates, add, out},
        mode_(add ? CNNL_SCATTERND_ADD : CNNL_SCATTERND_UPDATE) {
    if (indices.ndim() == 0 || input.shape() != out.shape() ||
        input.dtype() != out.dtype() || updates.dtype() != out.dtype() ||
        (indices.dtype() != DataType::kInt32 && indices.dtype() != DataType::kInt64) ||
        input.device() != out.device() || indices.device() != out.device() ||
        updates.device() != out.device() ||
        !input.IsContiguous() || !indices.IsContiguous() ||
        !updates.IsContiguous() || !out.IsContiguous())
      throw std::invalid_argument("Unsupported Cambricon ScatterNd tensors");
    width_ = indices.size(-1);
    if (width_ == 0 || width_ > input.ndim())
      throw std::invalid_argument("Cambricon ScatterNd has invalid coordinate width");
    const auto index_shape = indices.shape();
    const auto input_shape = input.shape();
    const auto update_shape = updates.shape();
    std::vector<int64_t> expected(index_shape.begin(), index_shape.end() - 1);
    expected.insert(expected.end(), input_shape.begin() + width_, input_shape.end());
    if (expected.size() != update_shape.size() ||
        !std::equal(expected.begin(), expected.end(), update_shape.begin()))
      throw std::invalid_argument("Cambricon ScatterNd has invalid update shape");
    if (indices.numel() == 0 || out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    input_ = Describe(input, false);
    indices_ = Describe(indices, indices.ndim() == 1);
    updates_ = Describe(updates, indices.ndim() == 1);
    output_ = Describe(out, false);
    dims_ = cnrt_utils::AllocateDeviceBuffer(width_ * sizeof(int64_t));
    normalized_ = cnrt_utils::AllocateDeviceBuffer(
        indices.numel() * (indices.dtype() == DataType::kInt64 ? sizeof(int64_t) : sizeof(int32_t)));
    if (!dims_ || !normalized_)
      throw std::runtime_error("Cambricon ScatterNd index allocation failed");
    std::vector<int64_t> dims(input_shape.begin(), input_shape.begin() + width_);
    if (cnrtMemcpy(dims_.get(), dims.data(), width_ * sizeof(int64_t),
                   cnrtMemcpyHostToDev) != cnrtSuccess)
      throw std::runtime_error("Cambricon ScatterNd coordinate shape upload failed");
  }

  void operator()(const Tensor input, const Tensor indices, const Tensor updates,
                  bool, Tensor out) const override {
    if (out.numel() == 0) return;
    auto queue = static_cast<cnrtQueue_t>(stream_);
    if (indices.numel() == 0) {
      if (input.data() != out.data() &&
          cnrtMemcpyAsync(out.data(), const_cast<void*>(input.data()),
                          out.numel() * out.element_size(), queue,
                          cnrtMemcpyDevToDev) != cnrtSuccess)
        throw std::runtime_error("Cambricon ScatterNd empty index copy failed");
      return;
    }
    NormalizeScatterCoordinates(indices.data(), normalized_.get(),
        static_cast<const int64_t*>(dims_.get()), indices.numel(), width_, indices.dtype(), queue);
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), queue));
    INFINI_OPS_CNNL_CHECK(cnnlScatterNd_v2(
        handle_.get(), mode_, indices_.get(), normalized_.get(), updates_.get(),
        updates.data(), input_.get(), input.data(), output_.get(), out.data()));
  }

 private:
  static cnnl_utils::TensorDescriptor Describe(const Tensor& tensor, bool prepend) {
    auto desc = cnnl_utils::CreateTensorDescriptor();
    const auto shape = tensor.shape();
    std::vector<int64_t> dims(prepend ? 1 : 0, 1);
    dims.insert(dims.end(), shape.begin(), shape.end());
    INFINI_OPS_CNNL_CHECK(cnnlSetTensorDescriptor_v2(
        desc.get(), CNNL_LAYOUT_ARRAY, cnnl_utils::GetDataType(tensor.dtype()),
        static_cast<int>(dims.size()), dims.data()));
    return desc;
  }

  cnnlScatterNdMode_t mode_;
  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_, indices_, updates_, output_;
  cnrt_utils::DeviceBuffer dims_, normalized_;
  size_t width_ = 0;
};

}  // namespace infini::ops

#endif
