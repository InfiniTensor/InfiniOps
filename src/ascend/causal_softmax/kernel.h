#ifndef INFINI_OPS_ASCEND_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_ASCEND_CAUSAL_SOFTMAX_KERNEL_H_

#include <limits>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_copy.h"
#include "aclnn_masked_fill_scalar.h"
#include "aclnn_softmax.h"
#include "ascend/common.h"
#include "ascend/workspace_pool_.h"
#include "base/causal_softmax.h"
#include "data_type.h"
#include "operator.h"

namespace infini::ops {

// Implements causal softmax via three ACLNN calls:
//   1. InplaceCopy(temp, input)   — stride-aware copy to contiguous temp
//   buffer.
//   2. InplaceMaskedFillScalar(temp, mask, -inf) — apply upper-triangle mask.
//   3. Softmax(temp, dim=-1, out) — softmax over the last dimension.
//
// The boolean causal mask is pre-computed and uploaded to device once in the
// constructor. Its shape (seq_len, total_seq_len) broadcasts over the batch.
template <>
class Operator<CausalSoftmax, Device::Type::kAscend> : public CausalSoftmax {
 public:
  Operator(const Tensor input, Tensor out)
      : CausalSoftmax(input, out), in_cache_(input), out_cache_(out) {
    // Compute temp buffer size — allocated lazily from pool in `operator()`.
    size_t n_elems = input.numel();
    size_t elem_bytes = kDataTypeToSize.at(dtype_);
    temp_size_ = n_elems * elem_bytes;

    // Build a contiguous Tensor descriptor — data pointer set on first use.
    Tensor temp_t{nullptr, input.shape(), input.dtype(), input.device()};
    temp_cache_ = ascend::AclTensorCache(temp_t);

    // Causal mask: mask[i][j] = 1 when position j must be masked for query i.
    // Shape (seq_len, total_seq_len) – broadcasts over the batch dimension.
    size_t mask_elems = seq_len_ * total_seq_len_;
    std::vector<uint8_t> mask_host(mask_elems, 0);

    for (size_t i = 0; i < seq_len_; ++i) {
      auto vis_end = static_cast<int64_t>(total_seq_len_ - seq_len_ + i);

      for (auto j = vis_end + 1; j < static_cast<int64_t>(total_seq_len_);
           ++j) {
        mask_host[i * total_seq_len_ + j] = 1;
      }
    }

    aclrtMalloc(&mask_buf_, mask_elems, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(mask_buf_, mask_elems, mask_host.data(), mask_elems,
                ACL_MEMCPY_HOST_TO_DEVICE);

    std::vector<int64_t> mshape = {static_cast<int64_t>(seq_len_),
                                   static_cast<int64_t>(total_seq_len_)};
    std::vector<int64_t> mstrides = {static_cast<int64_t>(total_seq_len_), 1};
    mask_tensor_ = aclCreateTensor(mshape.data(), mshape.size(), ACL_BOOL,
                                   mstrides.data(), 0, ACL_FORMAT_ND,
                                   mshape.data(), mshape.size(), mask_buf_);

    // Scalar -inf for the masked-fill step. aclCreateScalar stores the pointer
    // rather than copying, so neg_inf_storage_ must stay alive with the object.
    neg_inf_ = aclCreateScalar(&neg_inf_storage_, ACL_FLOAT);
    // Workspaces are allocated lazily on first operator() call.
  }

  ~Operator() {
    if (!ascend::isAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.
    in_cache_.release();
    out_cache_.release();
    temp_cache_.release();

    // `mask_tensor_` leaks with `fill_exec_` at shutdown (see `64c367c`).
    if (mask_buf_) aclrtFree(mask_buf_);
    if (neg_inf_) aclDestroyScalar(neg_inf_);
  }

  void operator()(const Tensor input, Tensor out) const override {
    auto t_in = in_cache_.get(const_cast<void*>(input.data()));
    auto t_out = out_cache_.get(out.data());
    auto stream = static_cast<aclrtStream>(stream_);

    // Obtain shared temp buffer from pool.
    auto& temp = ascend::GetWorkspacePool().Ensure(stream, temp_size_, "temp");
    auto t_temp = temp_cache_.get(temp.buf);

    // Step 1: copy input (possibly non-contiguous) into contiguous temp.
    if (!copy_exec_) {
      aclnnInplaceCopyGetWorkspaceSize(t_temp, t_in, &copy_ws_, &copy_exec_);
      aclSetAclOpExecutorRepeatable(copy_exec_);
    } else {
      aclSetInputTensorAddr(copy_exec_, 0, t_temp, temp.buf);
      aclSetInputTensorAddr(copy_exec_, 1, t_in,
                            const_cast<void*>(input.data()));
    }
    auto& copy_arena = ascend::GetWorkspacePool().Ensure(stream, copy_ws_);
    aclnnInplaceCopy(copy_arena.buf, copy_ws_, copy_exec_, stream);

    // Step 2: mask upper-triangle positions with -inf in-place.
    // `mask_tensor_` and `neg_inf_` have stable addresses — first-call only.
    if (!fill_exec_) {
      aclnnInplaceMaskedFillScalarGetWorkspaceSize(
          t_temp, mask_tensor_, neg_inf_, &fill_ws_, &fill_exec_);
      aclSetAclOpExecutorRepeatable(fill_exec_);
    }
    auto& fill_arena = ascend::GetWorkspacePool().Ensure(stream, fill_ws_);
    aclnnInplaceMaskedFillScalar(fill_arena.buf, fill_ws_, fill_exec_, stream);

    // Step 3: softmax over the last dimension -> out.
    if (!softmax_exec_) {
      constexpr int64_t kLastDim = -1;
      aclnnSoftmaxGetWorkspaceSize(t_temp, kLastDim, t_out, &softmax_ws_,
                                   &softmax_exec_);
      aclSetAclOpExecutorRepeatable(softmax_exec_);
    } else {
      aclSetOutputTensorAddr(softmax_exec_, 0, t_out, out.data());
    }
    auto& softmax_arena = ascend::GetWorkspacePool().Ensure(stream, softmax_ws_);
    aclnnSoftmax(softmax_arena.buf, softmax_ws_, softmax_exec_, stream);
  }

 private:
  mutable ascend::AclTensorCache in_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable ascend::AclTensorCache temp_cache_;

  float neg_inf_storage_ = -std::numeric_limits<float>::infinity();

  uint64_t temp_size_ = 0;

  void* mask_buf_ = nullptr;

  aclTensor* mask_tensor_ = nullptr;

  aclScalar* neg_inf_ = nullptr;

  mutable aclOpExecutor* copy_exec_ = nullptr;

  mutable uint64_t copy_ws_ = 0;

  mutable aclOpExecutor* fill_exec_ = nullptr;

  mutable uint64_t fill_ws_ = 0;

  mutable aclOpExecutor* softmax_exec_ = nullptr;

  mutable uint64_t softmax_ws_ = 0;
};

}  // namespace infini::ops

#endif
