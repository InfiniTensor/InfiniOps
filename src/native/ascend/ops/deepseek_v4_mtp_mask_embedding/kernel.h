#ifndef INFINI_OPS_ASCEND_DEEPSEEK_V4_MTP_MASK_EMBEDDING_KERNEL_H_
#define INFINI_OPS_ASCEND_DEEPSEEK_V4_MTP_MASK_EMBEDDING_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_eq_scalar.h"
#include "aclnnop/aclnn_masked_fill_scalar.h"
#include "base/deepseek_v4_mtp_mask_embedding.h"
#include "native/ascend/common.h"
#include "native/ascend/error_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<DeepseekV4MtpMaskEmbedding, Device::Type::kAscend, 0>
    : public DeepseekV4MtpMaskEmbedding {
 public:
  static bool supports(Tensor embedding, const Tensor positions) {
    return embedding.device().type() == Device::Type::kAscend &&
           embedding.dtype() == DataType::kBFloat16 &&
           embedding.ndim() == 3 && embedding.IsContiguous() &&
           positions.device() == embedding.device() &&
           (positions.dtype() == DataType::kInt32 ||
            positions.dtype() == DataType::kInt64) &&
           positions.IsContiguous() &&
           positions.numel() == embedding.shape()[0] * embedding.shape()[1];
  }

  Operator(Tensor embedding, const Tensor positions)
      : DeepseekV4MtpMaskEmbedding(embedding, positions),
        embedding_cache_(embedding),
        rows_(embedding.shape()[0]),
        cols_(embedding.shape()[1]) {
    if (!supports(embedding, positions)) {
      throw std::invalid_argument(
          "unsupported Ascend DeepseekV4MtpMaskEmbedding operands");
    }
    Tensor position_view{
        const_cast<void*>(positions.data()), {rows_, cols_, 1},
        positions.dtype(), positions.device()};
    positions_cache_ = ascend::AclTensorCache(position_view);
    mask_cache_ = ascend::AclTensorCache(
        {static_cast<int64_t>(rows_), static_cast<int64_t>(cols_), 1},
        ACL_BOOL, nullptr);

    auto* t_embedding = embedding_cache_.get(embedding.data());
    auto* t_positions = positions_cache_.get(
        const_cast<void*>(positions.data()));
    auto* t_mask = mask_cache_.get(nullptr);

    if (positions.dtype() == DataType::kInt64) {
      zero_i64_ = 0;
      position_zero_ = aclCreateScalar(&zero_i64_, ACL_INT64);
    } else {
      zero_i32_ = 0;
      position_zero_ = aclCreateScalar(&zero_i32_, ACL_INT32);
    }
    zero_f32_ = 0.0f;
    fill_zero_ = aclCreateScalar(&zero_f32_, ACL_FLOAT);
    if (!position_zero_ || !fill_zero_) {
      throw std::runtime_error(
          "DeepseekV4MtpMaskEmbedding scalar creation failed");
    }

    ascend::CheckAcl(aclnnEqScalarGetWorkspaceSize(
                         t_positions, position_zero_, t_mask,
                         &eq_workspace_size_, &eq_executor_),
                     "DeepseekV4MtpMaskEmbedding equal workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(eq_executor_),
                     "DeepseekV4MtpMaskEmbedding equal repeatable executor");
    ascend::CheckAcl(aclnnInplaceMaskedFillScalarGetWorkspaceSize(
                         t_embedding, t_mask, fill_zero_,
                         &fill_workspace_size_, &fill_executor_),
                     "DeepseekV4MtpMaskEmbedding fill workspace query");
    ascend::CheckAcl(aclSetAclOpExecutorRepeatable(fill_executor_),
                     "DeepseekV4MtpMaskEmbedding fill repeatable executor");
    scratch_size_ = rows_ * cols_;
    workspace_size_ = std::max(eq_workspace_size_, fill_workspace_size_);
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    embedding_cache_.release();
    positions_cache_.release();
    mask_cache_.release();
    if (position_zero_) aclDestroyScalar(position_zero_);
    if (fill_zero_) aclDestroyScalar(fill_zero_);
  }

  bool capture_safe() const override { return false; }

  void operator()(Tensor embedding,
                  const Tensor positions) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, scratch_size_ + workspace_size_,
        "deepseek_v4_mtp_mask_embedding");
    auto* mask = arena.buf;
    auto* workspace = static_cast<uint8_t*>(arena.buf) + scratch_size_;
    auto* t_embedding = embedding_cache_.get(embedding.data());
    auto* t_positions = positions_cache_.get(
        const_cast<void*>(positions.data()));
    auto* t_mask = mask_cache_.get(mask);

    ascend::CheckAcl(aclSetInputTensorAddr(
                         eq_executor_, 0, t_positions,
                         const_cast<void*>(positions.data())),
                     "DeepseekV4MtpMaskEmbedding equal input address");
    ascend::CheckAcl(aclSetOutputTensorAddr(
                         eq_executor_, 0, t_mask, mask),
                     "DeepseekV4MtpMaskEmbedding equal output address");
    ascend::CheckAcl(aclnnEqScalar(
                         workspace, eq_workspace_size_, eq_executor_, stream),
                     "DeepseekV4MtpMaskEmbedding equal execution");

    ascend::CheckAcl(aclSetInputTensorAddr(
                         fill_executor_, 0, t_embedding, embedding.data()),
                     "DeepseekV4MtpMaskEmbedding fill embedding address");
    ascend::CheckAcl(aclSetInputTensorAddr(
                         fill_executor_, 1, t_mask, mask),
                     "DeepseekV4MtpMaskEmbedding fill mask address");
    ascend::CheckAcl(aclnnInplaceMaskedFillScalar(
                         workspace, fill_workspace_size_, fill_executor_,
                         stream),
                     "DeepseekV4MtpMaskEmbedding fill execution");
  }

 private:
  mutable ascend::AclTensorCache embedding_cache_;
  mutable ascend::AclTensorCache positions_cache_;
  mutable ascend::AclTensorCache mask_cache_;
  size_t rows_{0};
  size_t cols_{0};
  uint64_t scratch_size_{0};
  uint64_t workspace_size_{0};
  uint64_t eq_workspace_size_{0};
  uint64_t fill_workspace_size_{0};
  mutable aclOpExecutor* eq_executor_{nullptr};
  mutable aclOpExecutor* fill_executor_{nullptr};
  aclScalar* position_zero_{nullptr};
  aclScalar* fill_zero_{nullptr};
  int64_t zero_i64_{0};
  int32_t zero_i32_{0};
  float zero_f32_{0.0f};
};

}  // namespace infini::ops

#endif
