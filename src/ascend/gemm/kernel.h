// TODO: Rewrite to use aclnnMatmul with per-call tensor descriptors (like
// add/rms_norm/swiglu) instead of per-instance stable buffers. Current
// approach allocates persistent device buffers per operator cache entry,
// which exhausts NPU memory at scale.
#ifndef INFINI_OPS_ASCEND_GEMM_KERNEL_H_
#define INFINI_OPS_ASCEND_GEMM_KERNEL_H_

#include <cstring>
#include <mutex>
#include <unordered_map>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_addmm.h"
#include "aclnnop/aclnn_baddbmm.h"
#include "ascend/device.h"
#include "base/gemm.h"
#include "data_type.h"
#include "operator.h"

namespace infini::ops {

namespace detail {

inline aclScalar* stableScalar(float v) {
  static std::mutex mu;
  static std::unordered_map<uint32_t, std::pair<float, aclScalar*>> reg;
  std::lock_guard<std::mutex> g(mu);
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof bits);
  auto [it, inserted] = reg.emplace(bits, std::make_pair(v, (aclScalar*)nullptr));
  if (inserted)
    it->second.second = aclCreateScalar(&it->second.first, ACL_FLOAT);
  return it->second.second;
}

inline aclTensor* buildAclTensorAt(const Tensor& t, void* data,
                                   bool transpose_last2 = false) {
  std::vector<int64_t> shape(t.shape().begin(), t.shape().end());
  std::vector<int64_t> strides(t.strides().begin(), t.strides().end());

  if (transpose_last2) {
    auto n = shape.size();
    std::swap(shape[n - 2], shape[n - 1]);
    std::swap(strides[n - 2], strides[n - 1]);
  }

  int64_t storage_elems = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) { storage_elems = 0; break; }
    if (strides[i] > 0 && shape[i] > 1)
      storage_elems += static_cast<int64_t>(shape[i] - 1) * strides[i];
  }
  std::vector<int64_t> storage_shape = {storage_elems};

  return aclCreateTensor(
      shape.data(), static_cast<int64_t>(shape.size()),
      ascend::toAclDtype(t.dtype()),
      strides.data(), 0, ACL_FORMAT_ND,
      storage_shape.data(), static_cast<int64_t>(storage_shape.size()),
      data);
}

inline size_t storageBytes(const Tensor& t) {
  int64_t storage_elems = 1;
  for (size_t i = 0; i < t.shape().size(); ++i) {
    if (t.shape()[i] == 0) { storage_elems = 0; break; }
    if (t.strides()[i] > 0 && t.shape()[i] > 1)
      storage_elems += static_cast<int64_t>(t.shape()[i] - 1) * t.strides()[i];
  }
  return static_cast<size_t>(storage_elems) * kDataTypeToSize.at(t.dtype());
}

}  // namespace detail

template <>
class Operator<Gemm, Device::Type::kAscend> : public Gemm {
 public:
  Operator(const Tensor a, const Tensor b, std::optional<float> alpha,
           std::optional<float> beta, std::optional<int> trans_a,
           std::optional<int> trans_b, Tensor c)
      : Gemm(a, b, alpha, beta, trans_a, trans_b, c),
        batched_{batch_count_ > 1},
        alpha_val_(alpha.value_or(1.0f)),
        beta_val_(beta.value_or(1.0f)) {
    auto stream = static_cast<aclrtStream>(stream_);
    a_bytes_ = detail::storageBytes(a);
    b_bytes_ = detail::storageBytes(b);
    c_bytes_ = detail::storageBytes(c);

    aclrtMalloc(&a_buf_, a_bytes_, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&b_buf_, b_bytes_, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&c_buf_, c_bytes_, ACL_MEM_MALLOC_NORMAL_ONLY);

    t_self_ = detail::buildAclTensorAt(c, c_buf_);
    t_a_    = detail::buildAclTensorAt(a, a_buf_, trans_a_);
    t_b_    = detail::buildAclTensorAt(b, b_buf_, trans_b_);

    auto* alpha_s = detail::stableScalar(alpha_val_);
    auto* beta_s  = detail::stableScalar(beta_val_);

    if (batched_) {
      aclnnInplaceBaddbmmGetWorkspaceSize(t_self_, t_a_, t_b_, beta_s,
                                          alpha_s, 0, &workspace_needed_,
                                          &executor_);
    } else {
      aclnnInplaceAddmmGetWorkspaceSize(t_self_, t_a_, t_b_, beta_s,
                                        alpha_s, 0, &workspace_needed_,
                                        &executor_);
    }
    aclSetAclOpExecutorRepeatable(executor_);

    if (workspace_needed_ > 0) {
      aclrtMalloc(&workspace_, workspace_needed_, ACL_MEM_MALLOC_NORMAL_ONLY);
      workspace_size_ = workspace_needed_;
    }
  }

  ~Operator() {
    aclDestroyTensor(t_self_);
    aclDestroyTensor(t_a_);
    aclDestroyTensor(t_b_);
    if (workspace_size_ > 0) aclrtFree(workspace_);
    if (a_buf_) aclrtFree(a_buf_);
    if (b_buf_) aclrtFree(b_buf_);
    if (c_buf_) aclrtFree(c_buf_);
  }

  void operator()(const Tensor a, const Tensor b, std::optional<float> alpha,
                  std::optional<float> beta, std::optional<int> trans_a,
                  std::optional<int> trans_b, Tensor c) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    // Copy caller data into stable buffers.
    aclrtMemcpy(a_buf_, a_bytes_, const_cast<void*>(a.data()), a_bytes_,
                ACL_MEMCPY_DEVICE_TO_DEVICE);
    aclrtMemcpy(b_buf_, b_bytes_, const_cast<void*>(b.data()), b_bytes_,
                ACL_MEMCPY_DEVICE_TO_DEVICE);
    aclrtMemcpy(c_buf_, c_bytes_, const_cast<void*>(c.data()), c_bytes_,
                ACL_MEMCPY_DEVICE_TO_DEVICE);

    if (batched_) {
      aclnnInplaceBaddbmm(workspace_, workspace_size_, executor_, stream);
    } else {
      aclnnInplaceAddmm(workspace_, workspace_size_, executor_, stream);
    }

    aclrtSynchronizeStream(stream);

    // Copy result back.
    aclrtMemcpy(const_cast<void*>(c.data()), c_bytes_, c_buf_, c_bytes_,
                ACL_MEMCPY_DEVICE_TO_DEVICE);
    aclrtSynchronizeStream(stream);
  }

 private:
  bool           batched_          = false;
  float          alpha_val_;
  float          beta_val_;
  size_t         a_bytes_          = 0;
  size_t         b_bytes_          = 0;
  size_t         c_bytes_          = 0;
  void*          a_buf_            = nullptr;
  void*          b_buf_            = nullptr;
  void*          c_buf_            = nullptr;
  aclTensor*     t_self_           = nullptr;
  aclTensor*     t_a_              = nullptr;
  aclTensor*     t_b_              = nullptr;
  aclOpExecutor* executor_         = nullptr;
  uint64_t       workspace_needed_ = 0;
  void*          workspace_        = nullptr;
  size_t         workspace_size_   = 0;
};

}  // namespace infini::ops

#endif
