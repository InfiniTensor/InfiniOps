#ifndef INFINI_OPS_CAMBRICON_SCATTER_KERNEL_H_
#define INFINI_OPS_CAMBRICON_SCATTER_KERNEL_H_

#include "base/scatter.h"
#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"
#include "native/cambricon/scatter_index.h"

namespace infini::ops {

template <>
class Operator<Scatter, Device::Type::kCambricon> : public Scatter {
 public:
  Operator(const Tensor input, const Tensor index, const Tensor src,
           const int64_t dim, Tensor out)
      : Operator{input, index, src, dim, "none", out} {}

  Operator(const Tensor input, const Tensor index, const Tensor src,
           const int64_t dim, const std::string reduce, Tensor out)
      : Scatter{input, index, src, dim, reduce, out}, mode_(Mode(reduce)) {
    const auto rank = static_cast<int64_t>(input.ndim());
    axis_ = dim < 0 ? dim + rank : dim;
    if (axis_ < 0 || axis_ >= rank || input.shape() != out.shape() ||
        index.shape() != src.shape() || input.ndim() != index.ndim() ||
        input.dtype() != out.dtype() || src.dtype() != out.dtype() ||
        (index.dtype() != DataType::kInt32 && index.dtype() != DataType::kInt64) ||
        input.device() != out.device() || index.device() != out.device() ||
        src.device() != out.device() || !input.IsContiguous() ||
        !out.IsContiguous() || (index.numel() && (!index.IsContiguous() || !src.IsContiguous())))
      throw std::invalid_argument("Unsupported Cambricon Scatter tensor metadata");
    for (int64_t i = 0; i < rank; ++i) {
      if (i != axis_ && index.size(i) > input.size(i))
        throw std::invalid_argument("Cambricon Scatter index extent exceeds input");
    }
    if (index.numel() == 0 || out.numel() == 0) return;
    handle_ = cnnl_utils::CreateHandle();
    input_ = cnnl_utils::MakeTensorDescriptor(input);
    index_ = cnnl_utils::MakeTensorDescriptor(index);
    source_ = cnnl_utils::MakeTensorDescriptor(src);
    output_ = cnnl_utils::MakeTensorDescriptor(out);
    normalized_ = cnrt_utils::AllocateDeviceBuffer(index.numel() * index.element_size());
    INFINI_OPS_CNNL_CHECK(cnnlGetScatterWorkspaceSize(
        handle_.get(), index_.get(), &workspace_size_));
    scratch_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
    if (!normalized_ || (workspace_size_ && !scratch_))
      throw std::runtime_error("Cambricon Scatter allocation failed");
  }

  Operator(const Tensor input, const Tensor index, const double value,
           const int64_t dim, Tensor out)
      : Scatter{input, index, value, dim, out} { UnsupportedScalar(); }
  Operator(const Tensor input, const Tensor index, const double value,
           const int64_t dim, const std::string reduce, Tensor out)
      : Scatter{input, index, value, dim, reduce, out} { UnsupportedScalar(); }

  std::size_t workspace_size_in_bytes() const override { return workspace_size_; }

  void operator()(const Tensor input, const Tensor index, const Tensor src,
                  const int64_t dim, Tensor out) const override {
    (*this)(input, index, src, dim, "none", out);
  }

  void operator()(const Tensor input, const Tensor index, const Tensor src,
                  const int64_t, const std::string, Tensor out) const override {
    if (out.numel() == 0) return;
    auto queue = static_cast<cnrtQueue_t>(stream_);
    if (index.numel() == 0) {
      if (input.data() != out.data() &&
          cnrtMemcpyAsync(out.data(), const_cast<void*>(input.data()),
                          out.numel() * out.element_size(), queue,
                          cnrtMemcpyDevToDev) != cnrtSuccess)
        throw std::runtime_error("Cambricon Scatter empty index copy failed");
      return;
    }
    NormalizeAxisIndices(index.data(), normalized_.get(), input.size(axis_),
                         index.numel(), index.dtype(), queue);
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), queue));
    auto* scratch = workspace_ && workspace_size_in_bytes_ >= workspace_size_
                        ? workspace_ : scratch_.get();
    INFINI_OPS_CNNL_CHECK(cnnlScatter_v2(
        handle_.get(), axis_, input_.get(), input.data(), index_.get(),
        normalized_.get(), source_.get(), src.data(), scratch, workspace_size_,
        output_.get(), out.data(), mode_));
  }

  void operator()(const Tensor, const Tensor, const double, const int64_t,
                  Tensor) const override { UnsupportedScalar(); }
  void operator()(const Tensor, const Tensor, const double, const int64_t,
                  const std::string, Tensor) const override { UnsupportedScalar(); }

 private:
  [[noreturn]] static void UnsupportedScalar() {
    throw std::invalid_argument("Cambricon Scatter requires tensor updates");
  }

  static cnnlScatterMode_t Mode(const std::string& reduce) {
    if (reduce.empty() || reduce == "none") return CNNL_SCATTER;
    if (reduce == "add") return CNNL_SCATTER_ADD;
    if (reduce == "max") return CNNL_SCATTER_MAX;
    if (reduce == "min") return CNNL_SCATTER_MIN;
    throw std::invalid_argument("Unsupported Cambricon Scatter reduction");
  }


  cnnlScatterMode_t mode_;
  int64_t axis_ = 0;
  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_, index_, source_, output_;
  cnrt_utils::DeviceBuffer normalized_, scratch_;
  std::size_t workspace_size_ = 0;
};

}  // namespace infini::ops

#endif
