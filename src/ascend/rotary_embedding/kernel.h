#ifndef INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_apply_rotary_pos_emb_v2.h"
#include "aclnnop/aclnn_index_select.h"
#include "aclnnop/aclnn_rotary_position_embedding.h"
#include "ascend/device.h"
#include "base/rotary_embedding.h"
#include "operator.h"

namespace infini::ops {

// aclnnApplyRotaryPosEmbV2 hardware constraints on Atlas A2/A3:
//   - rotaryMode "half" only (neox style)
//   - D (last dim of queryRef) must be 64 or 128
//   - bfloat16 only (float16 accumulates with ~1 ULP error that exceeds
//     atol=0.001 in tests; bfloat16 passes with atol=0.005)
//
// Use V2 when all three hold; fall back to V1 otherwise.
static bool use_rope_v2(int64_t D, bool is_neox, DataType dtype) {
    return is_neox && (D == 64 || D == 128) && dtype == DataType::kBFloat16;
}

template <>
class Operator<RotaryEmbedding, Device::Type::kAscend>
    : public RotaryEmbedding {
 public:
  Operator(const Tensor positions, const Tensor query, const Tensor key,
           const Tensor cos_sin_cache, int64_t head_size, int64_t rotary_dim,
           bool is_neox_style, Tensor query_out, Tensor key_out)
      : RotaryEmbedding(positions, query, key, cos_sin_cache, head_size,
                        rotary_dim, is_neox_style, query_out, key_out) {
    const int64_t max_seq_len = cos_sin_cache.size(0);
    const int64_t R = rotary_dim_;
    const int64_t half_R = R / 2;
    cache_elem_size_ = cos_sin_cache.element_size();

    // Copy raw cache to host for pre-expansion (one-time cost).
    size_t raw_bytes =
        static_cast<size_t>(max_seq_len * R) * cache_elem_size_;
    std::vector<uint8_t> cache_host(raw_bytes);
    aclrtMemcpy(cache_host.data(), raw_bytes, cos_sin_cache.data(),
                raw_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    // Pre-expand into separate cos/sin tables with duplicated values.
    // After expansion each row is R-wide:
    //   neox:      cos = [c0..c_{hR-1}, c0..c_{hR-1}]  (first half repeated)
    //   interleave: cos = [c0,c0, c1,c1, ..., c_{hR-1},c_{hR-1}]
    // Same pattern for sin.
    table_bytes_ = raw_bytes;
    std::vector<uint8_t> cos_table_host(table_bytes_);
    std::vector<uint8_t> sin_table_host(table_bytes_);

    for (int64_t p = 0; p < max_seq_len; ++p) {
      if (is_neox_style_) {
        for (int64_t j = 0; j < half_R; ++j) {
          const uint8_t* c_src =
              cache_host.data() +
              static_cast<size_t>(p * R + j) * cache_elem_size_;
          const uint8_t* s_src =
              cache_host.data() +
              static_cast<size_t>(p * R + half_R + j) * cache_elem_size_;
          auto* cos_dst = cos_table_host.data();
          auto* sin_dst = sin_table_host.data();
          std::memcpy(cos_dst +
                          static_cast<size_t>(p * R + j) * cache_elem_size_,
                      c_src, cache_elem_size_);
          std::memcpy(cos_dst +
                          static_cast<size_t>(p * R + half_R + j) *
                              cache_elem_size_,
                      c_src, cache_elem_size_);
          std::memcpy(sin_dst +
                          static_cast<size_t>(p * R + j) * cache_elem_size_,
                      s_src, cache_elem_size_);
          std::memcpy(sin_dst +
                          static_cast<size_t>(p * R + half_R + j) *
                              cache_elem_size_,
                      s_src, cache_elem_size_);
        }
      } else {
        for (int64_t j = 0; j < half_R; ++j) {
          const uint8_t* c_src =
              cache_host.data() +
              static_cast<size_t>(p * R + j) * cache_elem_size_;
          const uint8_t* s_src =
              cache_host.data() +
              static_cast<size_t>(p * R + half_R + j) * cache_elem_size_;
          auto* cos_dst = cos_table_host.data();
          auto* sin_dst = sin_table_host.data();
          std::memcpy(cos_dst +
                          static_cast<size_t>(p * R + 2 * j) * cache_elem_size_,
                      c_src, cache_elem_size_);
          std::memcpy(cos_dst +
                          static_cast<size_t>(p * R + 2 * j + 1) *
                              cache_elem_size_,
                      c_src, cache_elem_size_);
          std::memcpy(sin_dst +
                          static_cast<size_t>(p * R + 2 * j) * cache_elem_size_,
                      s_src, cache_elem_size_);
          std::memcpy(sin_dst +
                          static_cast<size_t>(p * R + 2 * j + 1) *
                              cache_elem_size_,
                      s_src, cache_elem_size_);
        }
      }
    }

    // Upload expanded tables to device (one-time).
    aclrtMalloc(&cos_table_dev_, table_bytes_, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&sin_table_dev_, table_bytes_, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(cos_table_dev_, table_bytes_, cos_table_host.data(),
                table_bytes_, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(sin_table_dev_, table_bytes_, sin_table_host.data(),
                table_bytes_, ACL_MEMCPY_HOST_TO_DEVICE);

    const int64_t T = num_tokens_;
    const int64_t Nq = num_heads_;
    const int64_t Nkv = num_kv_heads_;
    const int64_t D = head_size_;
    const bool v2 = use_rope_v2(R, is_neox_style_, query.dtype());
    use_v2_ = v2;

    // Gathered output buffers [T, R] — filled by aclnnIndexSelect at runtime.
    gathered_cs_bytes_ = static_cast<size_t>(T * R) * cache_elem_size_;
    aclrtMalloc(&cos_dev_, gathered_cs_bytes_, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&sin_dev_, gathered_cs_bytes_, ACL_MEM_MALLOC_NORMAL_ONLY);

    // Scratch for partial-rotation (R < D) — used by both V1 and V2.
    if (R < D) {
      size_t q_rot_bytes =
          static_cast<size_t>(T * Nq * R) * cache_elem_size_;
      size_t k_rot_bytes =
          static_cast<size_t>(T * Nkv * R) * cache_elem_size_;
      aclrtMalloc(&q_rot_dev_, q_rot_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
      aclrtMalloc(&k_rot_dev_, k_rot_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
      if (!v2) {
        aclrtMalloc(&q_out_rot_dev_, q_rot_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
        aclrtMalloc(&k_out_rot_dev_, k_rot_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
      }
    }
  }

  ~Operator() {
    if (cos_table_dev_) aclrtFree(cos_table_dev_);
    if (sin_table_dev_) aclrtFree(sin_table_dev_);
    if (cos_dev_) aclrtFree(cos_dev_);
    if (sin_dev_) aclrtFree(sin_dev_);
    if (q_rot_dev_) aclrtFree(q_rot_dev_);
    if (k_rot_dev_) aclrtFree(k_rot_dev_);
    if (q_out_rot_dev_) aclrtFree(q_out_rot_dev_);
    if (k_out_rot_dev_) aclrtFree(k_out_rot_dev_);
  }

  void operator()(const Tensor positions, const Tensor query, const Tensor key,
                  const Tensor cos_sin_cache, int64_t head_size,
                  int64_t rotary_dim, bool is_neox_style, Tensor query_out,
                  Tensor key_out) const override {
    auto stream = static_cast<aclrtStream>(stream_);

    const int64_t T = query.size(0);
    const int64_t Nq = query.size(1);
    const int64_t Nkv = key.size(1);
    const int64_t D = head_size;
    const int64_t R = rotary_dim;
    const int64_t max_seq_len = cos_sin_cache.size(0);

    assert(R <= D);
    assert(cos_sin_cache.size(1) == R);

    // 1. Gather cos/sin on device via aclnnIndexSelect — fully async.
    //    No host sync, no D2H copy. Positions stay on device.
    {
      aclDataType acl_dt_cs = ascend::toAclDtype(query.dtype());

      // Table tensors: [max_seq_len, R]
      std::vector<int64_t> table_shape = {max_seq_len, R};
      std::vector<int64_t> table_strides = {R, 1};
      std::vector<int64_t> table_storage = {max_seq_len * R};

      aclTensor* t_cos_table = aclCreateTensor(
          table_shape.data(), 2, acl_dt_cs, table_strides.data(), 0,
          ACL_FORMAT_ND, table_storage.data(), 1, cos_table_dev_);
      aclTensor* t_sin_table = aclCreateTensor(
          table_shape.data(), 2, acl_dt_cs, table_strides.data(), 0,
          ACL_FORMAT_ND, table_storage.data(), 1, sin_table_dev_);

      // Index tensor: positions [T], int64 — stays on device.
      std::vector<int64_t> idx_shape = {T};
      std::vector<int64_t> idx_strides = {1};
      std::vector<int64_t> idx_storage = {T};
      aclTensor* t_idx = aclCreateTensor(
          idx_shape.data(), 1, ACL_INT64, idx_strides.data(), 0,
          ACL_FORMAT_ND, idx_storage.data(), 1,
          const_cast<void*>(positions.data()));

      // Output tensors: [T, R]
      std::vector<int64_t> out_shape = {T, R};
      std::vector<int64_t> out_strides = {R, 1};
      std::vector<int64_t> out_storage = {T * R};

      aclTensor* t_cos_out = aclCreateTensor(
          out_shape.data(), 2, acl_dt_cs, out_strides.data(), 0,
          ACL_FORMAT_ND, out_storage.data(), 1, cos_dev_);
      aclTensor* t_sin_out = aclCreateTensor(
          out_shape.data(), 2, acl_dt_cs, out_strides.data(), 0,
          ACL_FORMAT_ND, out_storage.data(), 1, sin_dev_);

      // Get workspace sizes and executors for both gathers.
      uint64_t ws_cos = 0, ws_sin = 0;
      aclOpExecutor *exec_cos = nullptr, *exec_sin = nullptr;
      aclnnIndexSelectGetWorkspaceSize(t_cos_table, 0, t_idx, t_cos_out,
                                       &ws_cos, &exec_cos);
      aclnnIndexSelectGetWorkspaceSize(t_sin_table, 0, t_idx, t_sin_out,
                                       &ws_sin, &exec_sin);

      // Single workspace buffer large enough for both calls.
      uint64_t ws_max = ws_cos > ws_sin ? ws_cos : ws_sin;
      auto& arena = ascend::workspacePool().ensure(stream, ws_max);

      aclnnIndexSelect(arena.buf, ws_cos, exec_cos, stream);
      aclnnIndexSelect(arena.buf, ws_sin, exec_sin, stream);

      aclDestroyTensor(t_cos_table);
      aclDestroyTensor(t_sin_table);
      aclDestroyTensor(t_idx);
      aclDestroyTensor(t_cos_out);
      aclDestroyTensor(t_sin_out);
    }

    aclDataType acl_dt = ascend::toAclDtype(query.dtype());

    if (use_v2_) {
      // V2: fused Q+K, in-place, layout=4 (T-first 3D), "half" mode.
      // cos/sin shape: [T, 1, R].
      std::vector<int64_t> cs_shape = {T, 1, R};
      std::vector<int64_t> cs_strides = {R, R, 1};
      std::vector<int64_t> cs_storage = {T * R};
      aclTensor* t_cos =
          aclCreateTensor(cs_shape.data(), 3, acl_dt, cs_strides.data(), 0,
                          ACL_FORMAT_ND, cs_storage.data(), 1, cos_dev_);
      aclTensor* t_sin =
          aclCreateTensor(cs_shape.data(), 3, acl_dt, cs_strides.data(), 0,
                          ACL_FORMAT_ND, cs_storage.data(), 1, sin_dev_);

      int64_t layout = 4;
      if (R == D) {
        apply_rope_v2_full(query, key, query_out, key_out,
                           T, Nq, Nkv, D, acl_dt, t_cos, t_sin,
                           layout, stream);
      } else {
        apply_rope_v2_partial(query, key, query_out, key_out,
                              T, Nq, Nkv, D, R, acl_dt, t_cos, t_sin,
                              layout, stream);
      }
      aclDestroyTensor(t_cos);
      aclDestroyTensor(t_sin);
    } else {
      // V1: separate Q and K calls, non-in-place, [1,T,1,R] cos/sin.
      std::vector<int64_t> cs_shape = {1, T, 1, R};
      std::vector<int64_t> cs_strides = {T * R, R, R, 1};
      std::vector<int64_t> cs_storage = {T * R};
      aclTensor* t_cos =
          aclCreateTensor(cs_shape.data(), 4, acl_dt, cs_strides.data(), 0,
                          ACL_FORMAT_ND, cs_storage.data(), 1, cos_dev_);
      aclTensor* t_sin =
          aclCreateTensor(cs_shape.data(), 4, acl_dt, cs_strides.data(), 0,
                          ACL_FORMAT_ND, cs_storage.data(), 1, sin_dev_);

      int64_t mode = is_neox_style ? 0 : 1;
      apply_rope_v1(query, query_out, T, Nq, D, R, mode, t_cos, t_sin,
                    q_rot_dev_, q_out_rot_dev_, stream);
      apply_rope_v1(key, key_out, T, Nkv, D, R, mode, t_cos, t_sin,
                    k_rot_dev_, k_out_rot_dev_, stream);

      aclDestroyTensor(t_cos);
      aclDestroyTensor(t_sin);
    }
  }

 private:
  size_t cache_elem_size_ = 1;

  // Pre-expanded cos/sin tables on device: [max_seq_len, R].
  // Built once in the constructor with neox/interleave duplication.
  void* cos_table_dev_ = nullptr;
  void* sin_table_dev_ = nullptr;
  size_t table_bytes_ = 0;

  // true when V2 hardware constraints are met (neox, D∈{64,128}, bf16).
  bool use_v2_ = false;

  // Device buffers for gathered [T, R] cos/sin (shared by V1 and V2).
  void* cos_dev_ = nullptr;
  void* sin_dev_ = nullptr;
  size_t gathered_cs_bytes_ = 0;

  // Scratch for partial rotation (R < D).
  void* q_rot_dev_ = nullptr;
  void* k_rot_dev_ = nullptr;
  void* q_out_rot_dev_ = nullptr;
  void* k_out_rot_dev_ = nullptr;

  // --- V2 helpers (neox bf16, D∈{64,128}) ---

  void apply_rope_v2_full(const Tensor& q, const Tensor& k,
                          Tensor& q_out, Tensor& k_out,
                          int64_t T, int64_t Nq, int64_t Nkv, int64_t D,
                          aclDataType acl_dt,
                          aclTensor* t_cos, aclTensor* t_sin,
                          int64_t layout, aclrtStream stream) const {
    size_t elem_sz = q.element_size();
    if (q.data() != q_out.data()) {
      aclrtMemcpyAsync(const_cast<void*>(q_out.data()),
                       static_cast<size_t>(T * Nq * D) * elem_sz,
                       q.data(),
                       static_cast<size_t>(T * Nq * D) * elem_sz,
                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }
    if (k.data() != k_out.data()) {
      size_t k_elem_sz = k.element_size();
      aclrtMemcpyAsync(const_cast<void*>(k_out.data()),
                       static_cast<size_t>(T * Nkv * D) * k_elem_sz,
                       k.data(),
                       static_cast<size_t>(T * Nkv * D) * k_elem_sz,
                       ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }
    std::vector<int64_t> q_shape = {T, Nq, D};
    std::vector<int64_t> q_strides = {Nq * D, D, 1};
    std::vector<int64_t> q_storage = {T * Nq * D};
    std::vector<int64_t> k_shape = {T, Nkv, D};
    std::vector<int64_t> k_strides = {Nkv * D, D, 1};
    std::vector<int64_t> k_storage = {T * Nkv * D};
    aclTensor* t_q =
        aclCreateTensor(q_shape.data(), 3, acl_dt, q_strides.data(), 0,
                        ACL_FORMAT_ND, q_storage.data(), 1,
                        const_cast<void*>(q_out.data()));
    aclTensor* t_k =
        aclCreateTensor(k_shape.data(), 3, acl_dt, k_strides.data(), 0,
                        ACL_FORMAT_ND, k_storage.data(), 1,
                        const_cast<void*>(k_out.data()));
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclnnApplyRotaryPosEmbV2GetWorkspaceSize(
        t_q, t_k, t_cos, t_sin, layout,
        const_cast<char*>("half"), &ws, &exec);
    auto& arena = ascend::workspacePool().ensure(stream, ws);
    aclnnApplyRotaryPosEmbV2(arena.buf, ws, exec, stream);
    aclDestroyTensor(t_q);
    aclDestroyTensor(t_k);
  }

  void apply_rope_v2_partial(const Tensor& q, const Tensor& k,
                             Tensor& q_out, Tensor& k_out,
                             int64_t T, int64_t Nq, int64_t Nkv,
                             int64_t D, int64_t R,
                             aclDataType acl_dt,
                             aclTensor* t_cos, aclTensor* t_sin,
                             int64_t layout, aclrtStream stream) const {
    size_t elem_sz = q.element_size();
    size_t k_elem_sz = k.element_size();
    const int64_t pass = D - R;

    for (int64_t i = 0; i < T * Nq; ++i) {
      aclrtMemcpyAsync(
          static_cast<uint8_t*>(q_rot_dev_) +
              static_cast<size_t>(i * R) * elem_sz,
          static_cast<size_t>(R) * elem_sz,
          static_cast<const uint8_t*>(q.data()) +
              static_cast<size_t>(i * D) * elem_sz,
          static_cast<size_t>(R) * elem_sz,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }
    for (int64_t i = 0; i < T * Nkv; ++i) {
      aclrtMemcpyAsync(
          static_cast<uint8_t*>(k_rot_dev_) +
              static_cast<size_t>(i * R) * k_elem_sz,
          static_cast<size_t>(R) * k_elem_sz,
          static_cast<const uint8_t*>(k.data()) +
              static_cast<size_t>(i * D) * k_elem_sz,
          static_cast<size_t>(R) * k_elem_sz,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }
    std::vector<int64_t> qr_shape = {T, Nq, R};
    std::vector<int64_t> qr_strides = {Nq * R, R, 1};
    std::vector<int64_t> qr_storage = {T * Nq * R};
    std::vector<int64_t> kr_shape = {T, Nkv, R};
    std::vector<int64_t> kr_strides = {Nkv * R, R, 1};
    std::vector<int64_t> kr_storage = {T * Nkv * R};
    aclTensor* t_q_rot =
        aclCreateTensor(qr_shape.data(), 3, acl_dt, qr_strides.data(), 0,
                        ACL_FORMAT_ND, qr_storage.data(), 1, q_rot_dev_);
    aclTensor* t_k_rot =
        aclCreateTensor(kr_shape.data(), 3, acl_dt, kr_strides.data(), 0,
                        ACL_FORMAT_ND, kr_storage.data(), 1, k_rot_dev_);
    uint64_t ws = 0;
    aclOpExecutor* exec = nullptr;
    aclnnApplyRotaryPosEmbV2GetWorkspaceSize(
        t_q_rot, t_k_rot, t_cos, t_sin, layout,
        const_cast<char*>("half"), &ws, &exec);
    auto& arena = ascend::workspacePool().ensure(stream, ws);
    aclnnApplyRotaryPosEmbV2(arena.buf, ws, exec, stream);
    aclDestroyTensor(t_q_rot);
    aclDestroyTensor(t_k_rot);

    for (int64_t i = 0; i < T * Nq; ++i) {
      aclrtMemcpyAsync(
          static_cast<uint8_t*>(const_cast<void*>(q_out.data())) +
              static_cast<size_t>(i * D) * elem_sz,
          static_cast<size_t>(R) * elem_sz,
          static_cast<uint8_t*>(q_rot_dev_) +
              static_cast<size_t>(i * R) * elem_sz,
          static_cast<size_t>(R) * elem_sz,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      aclrtMemcpyAsync(
          static_cast<uint8_t*>(const_cast<void*>(q_out.data())) +
              static_cast<size_t>(i * D + R) * elem_sz,
          static_cast<size_t>(pass) * elem_sz,
          static_cast<const uint8_t*>(q.data()) +
              static_cast<size_t>(i * D + R) * elem_sz,
          static_cast<size_t>(pass) * elem_sz,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }
    for (int64_t i = 0; i < T * Nkv; ++i) {
      aclrtMemcpyAsync(
          static_cast<uint8_t*>(const_cast<void*>(k_out.data())) +
              static_cast<size_t>(i * D) * k_elem_sz,
          static_cast<size_t>(R) * k_elem_sz,
          static_cast<uint8_t*>(k_rot_dev_) +
              static_cast<size_t>(i * R) * k_elem_sz,
          static_cast<size_t>(R) * k_elem_sz,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      aclrtMemcpyAsync(
          static_cast<uint8_t*>(const_cast<void*>(k_out.data())) +
              static_cast<size_t>(i * D + R) * k_elem_sz,
          static_cast<size_t>(pass) * k_elem_sz,
          static_cast<const uint8_t*>(k.data()) +
              static_cast<size_t>(i * D + R) * k_elem_sz,
          static_cast<size_t>(pass) * k_elem_sz,
          ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }
  }

  // --- V1 helper (fallback for non-neox, fp16, or D not in {64,128}) ---

  void apply_rope_v1(const Tensor& x, Tensor& out,
                     int64_t T, int64_t N, int64_t D, int64_t R,
                     int64_t mode, aclTensor* t_cos, aclTensor* t_sin,
                     void* x_rot_dev, void* out_rot_dev,
                     aclrtStream stream) const {
    aclDataType acl_dt = ascend::toAclDtype(x.dtype());
    size_t elem_sz = x.element_size();

    if (R < D) {
      for (int64_t i = 0; i < T * N; ++i) {
        aclrtMemcpyAsync(
            static_cast<uint8_t*>(x_rot_dev) +
                static_cast<size_t>(i * R) * elem_sz,
            static_cast<size_t>(R) * elem_sz,
            static_cast<const uint8_t*>(x.data()) +
                static_cast<size_t>(i * D) * elem_sz,
            static_cast<size_t>(R) * elem_sz,
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      }
      std::vector<int64_t> rot_sh = {1, T, N, R};
      std::vector<int64_t> rot_st = {T * N * R, N * R, R, 1};
      std::vector<int64_t> rot_storage = {T * N * R};
      aclTensor* t_x_rot =
          aclCreateTensor(rot_sh.data(), 4, acl_dt, rot_st.data(), 0,
                          ACL_FORMAT_ND, rot_storage.data(), 1, x_rot_dev);
      aclTensor* t_out_rot =
          aclCreateTensor(rot_sh.data(), 4, acl_dt, rot_st.data(), 0,
                          ACL_FORMAT_ND, rot_storage.data(), 1, out_rot_dev);
      uint64_t ws = 0;
      aclOpExecutor* exec = nullptr;
      aclnnRotaryPositionEmbeddingGetWorkspaceSize(
          t_x_rot, t_cos, t_sin, mode, t_out_rot, &ws, &exec);
      auto& arena = ascend::workspacePool().ensure(stream, ws);
      aclnnRotaryPositionEmbedding(arena.buf, ws, exec, stream);

      const int64_t pass = D - R;
      for (int64_t i = 0; i < T * N; ++i) {
        aclrtMemcpyAsync(
            static_cast<uint8_t*>(const_cast<void*>(out.data())) +
                static_cast<size_t>(i * D) * elem_sz,
            static_cast<size_t>(R) * elem_sz,
            static_cast<uint8_t*>(out_rot_dev) +
                static_cast<size_t>(i * R) * elem_sz,
            static_cast<size_t>(R) * elem_sz,
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        aclrtMemcpyAsync(
            static_cast<uint8_t*>(const_cast<void*>(out.data())) +
                static_cast<size_t>(i * D + R) * elem_sz,
            static_cast<size_t>(pass) * elem_sz,
            static_cast<const uint8_t*>(x.data()) +
                static_cast<size_t>(i * D + R) * elem_sz,
            static_cast<size_t>(pass) * elem_sz,
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      }
      aclDestroyTensor(t_x_rot);
      aclDestroyTensor(t_out_rot);
    } else {
      std::vector<int64_t> full_sh = {1, T, N, D};
      std::vector<int64_t> full_st = {T * N * D, N * D, D, 1};
      std::vector<int64_t> full_storage = {T * N * D};
      aclTensor* t_x =
          aclCreateTensor(full_sh.data(), 4, acl_dt, full_st.data(), 0,
                          ACL_FORMAT_ND, full_storage.data(), 1,
                          const_cast<void*>(x.data()));
      aclTensor* t_out =
          aclCreateTensor(full_sh.data(), 4, acl_dt, full_st.data(), 0,
                          ACL_FORMAT_ND, full_storage.data(), 1,
                          const_cast<void*>(out.data()));
      uint64_t ws = 0;
      aclOpExecutor* exec = nullptr;
      aclnnRotaryPositionEmbeddingGetWorkspaceSize(
          t_x, t_cos, t_sin, mode, t_out, &ws, &exec);
      auto& arena = ascend::workspacePool().ensure(stream, ws);
      aclnnRotaryPositionEmbedding(arena.buf, ws, exec, stream);
      aclDestroyTensor(t_x);
      aclDestroyTensor(t_out);
    }
  }
};

}  // namespace infini::ops

#endif
