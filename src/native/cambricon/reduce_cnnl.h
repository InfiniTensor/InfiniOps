#ifndef INFINI_OPS_CAMBRICON_REDUCE_CNNL_H_
#define INFINI_OPS_CAMBRICON_REDUCE_CNNL_H_

#include <set>
#include <stdexcept>

#include "native/cambricon/cnnl_utils.h"
#include "native/cambricon/cnrt_utils.h"

namespace infini::ops::cnnl_utils {

class ReducePlan {
 public:
  ReducePlan(cnnlReduceOp_t mode, const Tensor& input,
             const std::vector<int64_t>& dim, bool keepdim, const Tensor& out) {
    const bool floating = input.dtype() == DataType::kFloat32 || input.dtype() == DataType::kFloat16;
    const bool integer_sum = mode == CNNL_REDUCE_ADD &&
        (input.dtype() == DataType::kInt32 || input.dtype() == DataType::kInt64);
    if (input.ndim() == 0 ||
        input.device() != out.device() || input.dtype() != out.dtype() ||
        !(floating || integer_sum) || !input.IsContiguous() || !out.IsContiguous())
      throw std::invalid_argument("Cambricon Reduce requires contiguous inputs and a supported dtype");
    const auto rank = static_cast<int64_t>(input.ndim());
    std::set<int> axes;
    if (dim.empty()) {
      for (int i = 0; i < rank; ++i) axes.insert(i);
    } else {
      for (auto axis : dim) {
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank || !axes.insert(static_cast<int>(axis)).second)
          throw std::invalid_argument("Cambricon Reduce: invalid axes");
      }
    }
    auto reduced_shape = Tensor::Shape(input.shape());
    Tensor::Shape expected;
    for (int i = 0; i < rank; ++i) {
      const bool reduce = axes.count(i) != 0;
      if (reduce) reduced_shape[i] = 1;
      if (keepdim || !reduce) expected.push_back(reduced_shape[i]);
    }
    if (expected != out.shape())
      throw std::invalid_argument("Cambricon Reduce: invalid output shape");
    if (out.numel() == 0) return;
    if (input.numel() == 0)
      throw std::invalid_argument("Cambricon Reduce: empty reduction not supported");
    Tensor::Strides reduced_strides(reduced_shape.size());
    Tensor::Stride stride = 1;
    for (size_t i = reduced_shape.size(); i-- > 0;) {
      reduced_strides[i] = stride;
      stride *= reduced_shape[i];
    }
    handle_ = cnnl_utils::CreateHandle();
    input_ = cnnl_utils::MakeTensorDescriptor(input.dtype(), Tensor::Shape(input.shape()),
                                                Tensor::Strides(input.strides()));
    output_ = cnnl_utils::MakeTensorDescriptor(out.dtype(), reduced_shape, reduced_strides);
    cnnlReduceDescriptor_t descriptor = nullptr;
    INFINI_OPS_CNNL_CHECK(cnnlCreateReduceDescriptor(&descriptor));
    descriptor_.reset(descriptor);
    std::vector<int> axis_list(axes.begin(), axes.end());
    INFINI_OPS_CNNL_CHECK(cnnlSetReduceDescriptor_v2(
        descriptor_.get(), axis_list.data(), axis_list.size(), mode,
        floating ? CNNL_DTYPE_FLOAT : cnnl_utils::GetDataType(input.dtype()),
        CNNL_PROPAGATE_NAN, CNNL_REDUCE_NO_INDICES,
        CNNL_32BIT_INDICES, 0.0));
    INFINI_OPS_CNNL_CHECK(cnnlGetReduceOpWorkspaceSize(
        handle_.get(), input_.get(), output_.get(), descriptor_.get(), &workspace_size_));
    workspace_own_ = cnrt_utils::AllocateDeviceBuffer(workspace_size_);
    if (workspace_size_ && !workspace_own_)
      throw std::runtime_error("Cambricon Reduce workspace allocation failed");
  }

  std::size_t WorkspaceSize() const { return workspace_size_; }

  void Run(const Tensor& input, Tensor out, cnrtQueue_t queue,
           void* workspace, size_t capacity) const {
    if (!output_) return;
    INFINI_OPS_CNNL_CHECK(cnnlSetQueue(handle_.get(), queue));
    auto* scratch = workspace && capacity >= workspace_size_
                        ? workspace : workspace_own_.get();
    INFINI_OPS_CNNL_CHECK(cnnlReduce(
        handle_.get(), descriptor_.get(), scratch, workspace_size_, nullptr,
        input_.get(), input.data(), 0, nullptr, nullptr, output_.get(), out.data()));
  }

 private:
  cnnl_utils::Handle handle_;
  cnnl_utils::TensorDescriptor input_, output_;
  std::unique_ptr<std::remove_pointer_t<cnnlReduceDescriptor_t>,
                  decltype(&cnnlDestroyReduceDescriptor)>
      descriptor_{nullptr, cnnlDestroyReduceDescriptor};
  cnrt_utils::DeviceBuffer workspace_own_;
  std::size_t workspace_size_ = 0;
};

}  // namespace infini::ops::cnnl_utils

#endif
