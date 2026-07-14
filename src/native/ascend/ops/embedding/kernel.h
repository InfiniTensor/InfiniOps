#ifndef INFINI_OPS_ASCEND_EMBEDDING_KERNEL_H_
#define INFINI_OPS_ASCEND_EMBEDDING_KERNEL_H_

#include <cassert>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_embedding.h"
#include "base/embedding.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Embedding, Device::Type::kAscend> : public Embedding {
 public:
  Operator(const Tensor weight, const Tensor indices, const int64_t padding_idx,
           const bool scale_grad_by_freq, const bool sparse, Tensor out)
      : Embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse,
                  out),
        indices_cache_(indices),
        weight_cache_(weight),
        out_cache_(out) {
    assert((weight_dtype_ == DataType::kFloat16 ||
            weight_dtype_ == DataType::kBFloat16 ||
            weight_dtype_ == DataType::kFloat32) &&
           "`Embedding`: Ascend path supports `float16`, `bfloat16`, and "
           "`float32` weights");
  }

  Operator(const Tensor weight, const Tensor indices, Tensor out)
      : Operator(weight, indices, -1, false, false, out) {}

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    indices_cache_.release();
    weight_cache_.release();
    out_cache_.release();
  }

  void operator()(const Tensor weight, const Tensor indices,
                  const int64_t /*padding_idx*/,
                  const bool /*scale_grad_by_freq*/, const bool /*sparse*/,
                  Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    auto t_weight = weight_cache_.get(const_cast<void*>(weight.data()));
    auto t_indices = indices_cache_.get(const_cast<void*>(indices.data()));
    auto t_out = out_cache_.get(out.data());

    if (!executor_) {
      auto ret = aclnnEmbeddingGetWorkspaceSize(t_weight, t_indices, t_out,
                                                &ws_size_, &executor_);
      assert(ret == ACL_SUCCESS && "`aclnnEmbeddingGetWorkspaceSize` failed");
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      aclSetInputTensorAddr(executor_, 0, t_weight,
                            const_cast<void*>(weight.data()));
      aclSetInputTensorAddr(executor_, 1, t_indices,
                            const_cast<void*>(indices.data()));
      aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);
    auto ret = aclnnEmbedding(arena.buf, ws_size_, executor_, stream);
    assert(ret == ACL_SUCCESS && "`aclnnEmbedding` failed");
  }

 private:
  mutable ascend::AclTensorCache indices_cache_;

  mutable ascend::AclTensorCache weight_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_EMBEDDING_KERNEL_H_
