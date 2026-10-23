#ifndef INFINI_OPS_ASCEND_ROTARY_PACKED_CACHE_FALLBACK_INC_
#define INFINI_OPS_ASCEND_ROTARY_PACKED_CACHE_FALLBACK_INC_

#include <cassert>
#include <cstdint>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_copy.h"
#include "aclnnop/level2/aclnn_rope_with_sin_cos_cache.h"
#include "base/rotary_embedding.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {
// Pack strided inputs into independently owned contiguous buffers.
// Query-only operation duplicates query into the independent key input buffer;
// The resulting dummy key output is discarded; all descriptors and scratch
// buffers remain owned by this operator through asynchronous submission.
template <>
class Operator<RotaryEmbedding, Device::Type::kAscend>
    : public RotaryEmbedding {
 public:
  Operator(const Tensor positions, Tensor query, std::optional<Tensor> key,
           const Tensor cache, int64_t head_size, bool is_neox,
           int64_t offset = 0, bool inverse = false)
      : RotaryEmbedding(positions, query, key, cache, head_size, is_neox,
                        offset, inverse),
        has_key_(key.has_value()) {
    assert(rope_dim_offset_ == 0 && !inverse_ && rot_dim_ <= head_size_ &&
           (!is_neox_ || rot_dim_ == head_size_));
    assert(cache.dtype() == query.dtype() && cache.IsContiguous());
    const auto tokens = static_cast<int64_t>(num_tokens_);
    const auto heads = static_cast<int64_t>(num_heads_);
    const auto kv_heads = static_cast<int64_t>(num_kv_heads_);
    const auto dtype = ascend::ToAclDtype(query.dtype());
    positions_cache_ = ascend::AclTensorCache(
        {tokens}, ACL_INT64, const_cast<void*>(positions.data()));
    cache_cache_ = ascend::AclTensorCache(cache);
    // Preserve batch boundaries while packing; flatten only owned buffers.
    query_cache_ = ascend::AclTensorCache(query);
    auto ret = aclrtMalloc(&query_out_, query.numel() * query.element_size(),
                           ACL_MEM_MALLOC_NORMAL_ONLY);
    assert(ret == ACL_SUCCESS);
    query_out_cache_ =
        ascend::AclTensorCache({tokens, heads * head_size_}, dtype, query_out_);
    ret = aclrtMalloc(&query_in_, query.numel() * query.element_size(),
                      ACL_MEM_MALLOC_NORMAL_ONLY);
    assert(ret == ACL_SUCCESS);
    query_in_cache_ =
        ascend::AclTensorCache({tokens, heads * head_size_}, dtype, query_in_);
    const std::vector<int64_t> query_shape(query.shape().begin(),
                                           query.shape().end());
    query_pack_cache_ = ascend::AclTensorCache(query_shape, dtype, query_in_);
    query_unpack_cache_ =
        ascend::AclTensorCache(query_shape, dtype, query_out_);
    Tensor key_input = key.has_value() ? *key : query;
    key_cache_ = ascend::AclTensorCache(key_input);
    ret = aclrtMalloc(&key_out_, key_input.numel() * query.element_size(),
                      ACL_MEM_MALLOC_NORMAL_ONLY);
    assert(ret == ACL_SUCCESS);
    key_out_cache_ = ascend::AclTensorCache({tokens, kv_heads * head_size_},
                                            dtype, key_out_);
    ret = aclrtMalloc(&key_in_, key_input.numel() * query.element_size(),
                      ACL_MEM_MALLOC_NORMAL_ONLY);
    assert(ret == ACL_SUCCESS);
    key_in_cache_ =
        ascend::AclTensorCache({tokens, kv_heads * head_size_}, dtype, key_in_);
    const std::vector<int64_t> key_shape(key_input.shape().begin(),
                                         key_input.shape().end());
    key_pack_cache_ = ascend::AclTensorCache(key_shape, dtype, key_in_);
    key_unpack_cache_ = ascend::AclTensorCache(key_shape, dtype, key_out_);
  }
  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;
    positions_cache_.release();
    cache_cache_.release();
    query_cache_.release();
    key_cache_.release();
    query_out_cache_.release();
    key_out_cache_.release();
    query_in_cache_.release();
    key_in_cache_.release();
    query_pack_cache_.release();
    key_pack_cache_.release();
    query_unpack_cache_.release();
    key_unpack_cache_.release();
    if (query_in_) aclrtFree(query_in_);
    if (key_in_) aclrtFree(key_in_);
    if (query_out_) aclrtFree(query_out_);
    if (key_out_) aclrtFree(key_out_);
  }
  void operator()(const Tensor positions, Tensor query,
                  std::optional<Tensor> key, const Tensor cache,
                  int64_t head_size, bool is_neox, int64_t offset = 0,
                  bool inverse = false) const override {
    assert(key.has_value() == has_key_);
    assert(head_size == head_size_ && is_neox == is_neox_ &&
           offset == rope_dim_offset_ && inverse == inverse_);
    auto stream = static_cast<aclrtStream>(stream_);
    auto p = positions_cache_.get(const_cast<void*>(positions.data()));
    auto c = cache_cache_.get(const_cast<void*>(cache.data()));
    auto q = query_cache_.get(query.data());
    auto k = key_cache_.get(key.has_value() ? key->data() : query.data());
    auto qo = query_out_cache_.get(query_out_);
    auto ko = key_out_cache_.get(key_out_);
    CopyBack(query_pack_cache_.get(query_in_), q, stream);
    CopyBack(key_pack_cache_.get(key_in_), k, stream);
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnRopeWithSinCosCacheGetWorkspaceSize(
        p, query_in_cache_.get(query_in_), key_in_cache_.get(key_in_), c,
        nullptr, head_size_, is_neox_, qo, ko, &bytes, &executor);
    assert(ret == ACL_SUCCESS);
    auto& arena = ascend::GetWorkspacePool().Ensure(stream, bytes);
    ret = aclnnRopeWithSinCosCache(arena.buf, bytes, executor, stream);
    assert(ret == ACL_SUCCESS);
    CopyBack(q, query_unpack_cache_.get(query_out_), stream);
    if (key.has_value()) CopyBack(k, key_unpack_cache_.get(key_out_), stream);
  }

 private:
  static void CopyBack(aclTensor* dst, aclTensor* src, aclrtStream stream) {
    uint64_t bytes = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnInplaceCopyGetWorkspaceSize(dst, src, &bytes, &executor);
    assert(ret == ACL_SUCCESS);
    auto& arena = ascend::GetWorkspacePool().Ensure(stream, bytes);
    ret = aclnnInplaceCopy(arena.buf, bytes, executor, stream);
    assert(ret == ACL_SUCCESS);
  }
  bool has_key_;
  void* query_in_{nullptr};
  void* key_in_{nullptr};
  void* query_out_{nullptr};
  void* key_out_{nullptr};
  mutable ascend::AclTensorCache positions_cache_, cache_cache_;
  mutable ascend::AclTensorCache query_cache_, key_cache_;
  mutable ascend::AclTensorCache query_out_cache_, key_out_cache_;
  mutable ascend::AclTensorCache query_in_cache_, key_in_cache_;
  mutable ascend::AclTensorCache query_pack_cache_, key_pack_cache_;
  mutable ascend::AclTensorCache query_unpack_cache_, key_unpack_cache_;
};
}  // namespace infini::ops
#endif
