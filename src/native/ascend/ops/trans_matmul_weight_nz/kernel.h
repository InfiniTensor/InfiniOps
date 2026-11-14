#ifndef INFINI_OPS_ASCEND_TRANS_MATMUL_WEIGHT_NZ_KERNEL_H_
#define INFINI_OPS_ASCEND_TRANS_MATMUL_WEIGHT_NZ_KERNEL_H_

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_trans_matmul_weight.h"
#include "base/trans_matmul_weight_nz.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {
namespace ascend_trans_matmul_weight_nz_detail {

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

}  // namespace ascend_trans_matmul_weight_nz_detail

template <>
class Operator<TransMatmulWeightNz, Device::Type::kAscend>
    : public TransMatmulWeightNz {
 public:
  Operator(const Tensor input, Tensor output)
      : TransMatmulWeightNz(input, output) {}

  void operator()(const Tensor input, Tensor output) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    const auto bytes = input.numel() * kDataTypeToSize.at(input.dtype());
    ascend::CheckAcl(aclrtMemcpyAsync(output.data(), bytes, input.data(), bytes,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE, stream),
                     "TransMatmulWeightNz source copy");

    auto weight = ascend::OwnAclTensor(
        ascend_trans_matmul_weight_nz_detail::MakeContiguousTensor(output));
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    ascend::CheckAcl(aclnnTransMatmulWeightGetWorkspaceSize(
                         weight.get(), &workspace_size, &executor),
                     "TransMatmulWeightNz workspace query");

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size);
    ascend::CheckAcl(
        aclnnTransMatmulWeight(arena.buf, workspace_size, executor, stream),
        "TransMatmulWeightNz execution");
    ascend::CheckAcl(aclrtSynchronizeStream(stream),
                     "TransMatmulWeightNz synchronize");
  }
};

}  // namespace infini::ops

#endif
