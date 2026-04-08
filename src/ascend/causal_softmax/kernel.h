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
  Operator(const Tensor input, Tensor out) : CausalSoftmax(input, out) {
    // Contiguous temp buffer with the same element count as input.
    size_t n_elems = input.numel();
    size_t elem_bytes = kDataTypeToSize.at(dtype_);
    aclrtMalloc(&temp_buf_, n_elems * elem_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

    // Build a contiguous Tensor descriptor pointing to temp_buf_.
    Tensor temp_t{temp_buf_, input.shape(), input.dtype(), input.device()};

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
    aclrtFree(temp_buf_);
    aclrtFree(mask_buf_);
    aclDestroyTensor(mask_tensor_);
    aclDestroyScalar(neg_inf_);
  }

  void operator()(const Tensor input, Tensor out) const override {
    Tensor temp_t{temp_buf_, input.shape(), input.dtype(), input.device()};
    auto t_in = ascend::buildAclTensor(input);
    auto t_temp = ascend::buildAclTensor(temp_t);
    auto t_out = ascend::buildAclTensor(out);
    auto stream = static_cast<aclrtStream>(stream_);

    uint64_t ws_needed = 0;
    aclOpExecutor* exec = nullptr;

    // Step 1: copy input (possibly non-contiguous) into contiguous temp.
    aclnnInplaceCopyGetWorkspaceSize(t_temp, t_in, &ws_needed, &exec);
    auto& copy_arena = ascend::workspacePool().ensure(stream, ws_needed);
    uint64_t copy_ws = ws_needed;
    aclnnInplaceCopy(copy_arena.buf, copy_ws, exec, stream);

    // Step 2: mask upper-triangle positions with -inf in-place.
    ws_needed = 0;
    exec = nullptr;
    aclnnInplaceMaskedFillScalarGetWorkspaceSize(t_temp, mask_tensor_, neg_inf_,
                                                 &ws_needed, &exec);
    auto& fill_arena = ascend::workspacePool().ensure(stream, ws_needed);
    uint64_t fill_ws = ws_needed;
    aclnnInplaceMaskedFillScalar(fill_arena.buf, fill_ws, exec, stream);

    // Step 3: softmax over the last dimension → out.
    ws_needed = 0;
    exec = nullptr;
    constexpr int64_t kLastDim = -1;
    aclnnSoftmaxGetWorkspaceSize(t_temp, kLastDim, t_out, &ws_needed, &exec);
    auto& softmax_arena = ascend::workspacePool().ensure(stream, ws_needed);
    uint64_t softmax_ws = ws_needed;
    aclnnSoftmax(softmax_arena.buf, softmax_ws, exec, stream);

    aclDestroyTensor(t_in);
    aclDestroyTensor(t_temp);
    aclDestroyTensor(t_out);
  }

 private:
  float neg_inf_storage_ = -std::numeric_limits<float>::infinity();
  void* temp_buf_ = nullptr;
  void* mask_buf_ = nullptr;
  aclTensor* mask_tensor_ = nullptr;
  aclScalar* neg_inf_ = nullptr;
};

}  // namespace infini::ops

#endif
