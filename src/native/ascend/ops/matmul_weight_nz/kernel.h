#ifndef INFINI_OPS_ASCEND_MATMUL_WEIGHT_NZ_KERNEL_H_
#define INFINI_OPS_ASCEND_MATMUL_WEIGHT_NZ_KERNEL_H_

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_matmul.h"
#include "base/matmul_weight_nz.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {
namespace ascend_matmul_weight_nz_detail {

inline aclTensor* MakeContiguousTensor(const Tensor& tensor) {
  std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
  std::vector<int64_t> strides(tensor.strides().begin(),
                               tensor.strides().end());
  std::vector<int64_t> storage{static_cast<int64_t>(tensor.numel())};
  return aclCreateTensor(shape.data(), static_cast<int64_t>(shape.size()),
                         ascend::ToAclDtype(tensor.dtype()), strides.data(), 0,
                         ACL_FORMAT_ND, storage.data(),
                         static_cast<int64_t>(storage.size()),
                         const_cast<void*>(tensor.data()));
}

inline aclTensor* MakeNzWeight(const Tensor& weight) {
  const int64_t rows = static_cast<int64_t>(weight.size(0));
  const int64_t columns = static_cast<int64_t>(weight.size(1));
  std::vector<int64_t> shape{rows, columns};
  std::vector<int64_t> strides{columns, 1};
  std::vector<int64_t> storage{columns / 16, rows / 16, 16, 16};
  return aclCreateTensor(shape.data(), static_cast<int64_t>(shape.size()),
                         ascend::ToAclDtype(weight.dtype()), strides.data(), 0,
                         ACL_FORMAT_FRACTAL_NZ, storage.data(),
                         static_cast<int64_t>(storage.size()),
                         const_cast<void*>(weight.data()));
}

}  // namespace ascend_matmul_weight_nz_detail

template <>
class Operator<MatmulWeightNz, Device::Type::kAscend> : public MatmulWeightNz {
 public:
  Operator(const Tensor input, const Tensor weight, Tensor out)
      : MatmulWeightNz(input, weight, out) {}

  void operator()(const Tensor input, const Tensor weight,
                  Tensor out) const override {
    using ascend_matmul_weight_nz_detail::MakeContiguousTensor;
    using ascend_matmul_weight_nz_detail::MakeNzWeight;

    auto input_tensor = ascend::OwnAclTensor(MakeContiguousTensor(input));
    auto weight_tensor = ascend::OwnAclTensor(MakeNzWeight(weight));
    auto out_tensor = ascend::OwnAclTensor(MakeContiguousTensor(out));

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    ascend::CheckAcl(aclnnMatmulWeightNzGetWorkspaceSize(
                         input_tensor.get(), weight_tensor.get(),
                         out_tensor.get(), 1, &workspace_size, &executor),
                     "MatmulWeightNz workspace query");

    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size);
    ascend::CheckAcl(
        aclnnMatmulWeightNz(arena.buf, workspace_size, executor, stream),
        "MatmulWeightNz execution");
  }
};

}  // namespace infini::ops

#endif
