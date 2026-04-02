#ifndef INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_rotary_position_embedding.h"
#include "ascend/device.h"
#include "base/rotary_embedding.h"
#include "operator.h"

namespace infini::ops {

namespace detail {

// Build an aclTensor with a different view shape/stride but the same data
// pointer as the source Tensor.
inline aclTensor* reshapeAclView(const Tensor& t,
                                 const std::vector<int64_t>& new_shape,
                                 const std::vector<int64_t>& new_strides) {
  int64_t storage_elems = 1;
  for (size_t i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] == 0) {
      storage_elems = 0;
      break;
    }
    if (new_strides[i] > 0 && new_shape[i] > 1) {
      storage_elems += static_cast<int64_t>(new_shape[i] - 1) * new_strides[i];
    }
  }
  std::vector<int64_t> storage_shape = {storage_elems};
  return aclCreateTensor(
      new_shape.data(), static_cast<int64_t>(new_shape.size()),
      ascend::toAclDtype(t.dtype()), new_strides.data(), 0, ACL_FORMAT_ND,
      storage_shape.data(), static_cast<int64_t>(storage_shape.size()),
      const_cast<void*>(t.data()));
}

}  // namespace detail

template <>
class Operator<RotaryEmbedding, Device::Type::kAscend>
    : public RotaryEmbedding {
 public:
  using RotaryEmbedding::RotaryEmbedding;

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
    const int64_t half_R = R / 2;

    assert(R <= D);
    assert(cos_sin_cache.size(1) == R);

    // mode: 0=half (neox), 1=interleave (non-neox).
    int64_t mode = is_neox_style_ ? 0 : 1;

    // 1. Copy positions to host for cos/sin gathering.
    std::vector<int64_t> pos(T);
    aclrtMemcpyAsync(pos.data(), T * sizeof(int64_t), positions.data(),
                     T * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    aclrtSynchronizeStream(stream);

    // 2. Copy cos_sin_cache to host: [max_seq_len, R].
    //    Layout: first half_R cols = cos, last half_R cols = sin.
    auto cache_n = cos_sin_cache.numel();
    auto elem_sz = cos_sin_cache.element_size();
    std::vector<uint8_t> cache_host(cache_n * elem_sz);
    aclrtMemcpyAsync(cache_host.data(), cache_host.size(),
                     cos_sin_cache.data(), cache_host.size(),
                     ACL_MEMCPY_DEVICE_TO_HOST, stream);
    aclrtSynchronizeStream(stream);

    // 3. Build device-side cos [1,T,1,R] and sin [1,T,1,R].
    //    Gather cos/sin rows from cache, expand to R by duplicating half_R.
    //    aclnnRotaryPositionEmbedding mode=0 (half/neox) does:
    //      out = cos * x + sin * rotate_half(x)
    //    where rotate_half(x) = [-x2, x1] for x = [x1, x2].
    //    cos/sin need to be [cos_half, cos_half] / [sin_half, sin_half].
    std::vector<float> cos_buf(T * R);
    std::vector<float> sin_buf(T * R);

    auto read_val = [&](int64_t flat_idx) -> float {
      auto off = flat_idx * elem_sz;
      switch (cos_sin_cache.dtype()) {
        case DataType::kFloat16: {
          Float16 v;
          std::memcpy(&v, &cache_host[off], sizeof(Float16));
          return v.ToFloat();
        }
        case DataType::kBFloat16: {
          BFloat16 v;
          std::memcpy(&v, &cache_host[off], sizeof(BFloat16));
          return v.ToFloat();
        }
        case DataType::kFloat32: {
          float v;
          std::memcpy(&v, &cache_host[off], sizeof(float));
          return v;
        }
        default:
          assert(false && "RotaryEmbedding: unsupported cos_sin_cache dtype");
          return 0.0f;
      }
    };

    for (int64_t t = 0; t < T; ++t) {
      int64_t p = pos[t];
      for (int64_t j = 0; j < half_R; ++j) {
        float c = read_val(p * R + j);
        float s = read_val(p * R + half_R + j);
        if (is_neox_style) {
          // mode=0 (half): [c0, c1, ..., c_{R/2-1}, c0, c1, ..., c_{R/2-1}]
          cos_buf[t * R + j] = c;
          cos_buf[t * R + half_R + j] = c;
          sin_buf[t * R + j] = s;
          sin_buf[t * R + half_R + j] = s;
        } else {
          // mode=2 (interleave): [c0, c0, c1, c1, ..., c_{R/2-1}, c_{R/2-1}]
          cos_buf[t * R + 2 * j] = c;
          cos_buf[t * R + 2 * j + 1] = c;
          sin_buf[t * R + 2 * j] = s;
          sin_buf[t * R + 2 * j + 1] = s;
        }
      }
    }

    // Allocate and upload cos/sin to device in query's dtype.
    size_t cs_bytes = T * R * query.element_size();
    void* cos_dev = nullptr;
    void* sin_dev = nullptr;
    aclrtMalloc(&cos_dev, cs_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMalloc(&sin_dev, cs_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

    auto upload = [&](const std::vector<float>& src, void* dev) {
      switch (query.dtype()) {
        case DataType::kFloat16: {
          std::vector<Float16> tmp(T * R);
          for (size_t i = 0; i < tmp.size(); ++i)
            tmp[i] = Float16::FromFloat(src[i]);
          aclrtMemcpyAsync(dev, cs_bytes, tmp.data(), cs_bytes,
                           ACL_MEMCPY_HOST_TO_DEVICE, stream);
          break;
        }
        case DataType::kBFloat16: {
          std::vector<BFloat16> tmp(T * R);
          for (size_t i = 0; i < tmp.size(); ++i)
            tmp[i] = BFloat16::FromFloat(src[i]);
          aclrtMemcpyAsync(dev, cs_bytes, tmp.data(), cs_bytes,
                           ACL_MEMCPY_HOST_TO_DEVICE, stream);
          break;
        }
        case DataType::kFloat32: {
          aclrtMemcpyAsync(dev, cs_bytes, src.data(), cs_bytes,
                           ACL_MEMCPY_HOST_TO_DEVICE, stream);
          break;
        }
        default:
          assert(false && "RotaryEmbedding: unsupported query dtype");
      }
    };
    upload(cos_buf, cos_dev);
    upload(sin_buf, sin_dev);
    aclrtSynchronizeStream(stream);

    // Build aclTensors for cos/sin: [1, T, 1, R].
    aclDataType acl_dt = ascend::toAclDtype(query.dtype());
    std::vector<int64_t> cs_shape = {1, T, 1, R};
    std::vector<int64_t> cs_strides = {T * R, R, R, 1};
    std::vector<int64_t> cs_storage = {T * R};
    aclTensor* t_cos = aclCreateTensor(cs_shape.data(), 4, acl_dt,
                                        cs_strides.data(), 0, ACL_FORMAT_ND,
                                        cs_storage.data(), 1, cos_dev);
    aclTensor* t_sin = aclCreateTensor(cs_shape.data(), 4, acl_dt,
                                        cs_strides.data(), 0, ACL_FORMAT_ND,
                                        cs_storage.data(), 1, sin_dev);

    // 4. Apply RoPE via aclnnRotaryPositionEmbedding.
    //    Input: [1, T, N, D] (4D).  If R < D, only rotate first R dims.
    auto apply_rope = [&](const Tensor& x, Tensor& out, int64_t N) {
      size_t elem_sz = x.element_size();

      if (R < D) {
        // CANN's RoPE ignores output strides and writes contiguously, so we
        // cannot use a strided view of `out` as the output.  Instead:
        //   1. Gather the rotary part of x into a contiguous temp buffer.
        //   2. Run CANN RoPE on the contiguous buffer.
        //   3. Scatter the result back to out.
        //   4. Copy the non-rotary part from x to out directly.

        size_t rot_bytes = static_cast<size_t>(T * N * R) * elem_sz;
        void* x_rot_dev = nullptr;
        void* out_rot_dev = nullptr;
        aclrtMalloc(&x_rot_dev, rot_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
        aclrtMalloc(&out_rot_dev, rot_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

        // Gather: copy first R elements per row from x to contiguous buffer.
        for (int64_t i = 0; i < T * N; ++i) {
          aclrtMemcpyAsync(
              static_cast<uint8_t*>(x_rot_dev) + i * R * elem_sz,
              static_cast<size_t>(R) * elem_sz,
              static_cast<const uint8_t*>(x.data()) + i * D * elem_sz,
              static_cast<size_t>(R) * elem_sz,
              ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        }

        // Contiguous [1, T, N, R] for CANN.
        std::vector<int64_t> cont_sh = {1, T, N, R};
        std::vector<int64_t> cont_st = {T * N * R, N * R, R, 1};
        std::vector<int64_t> cont_storage = {T * N * R};
        aclTensor* t_x_rot = aclCreateTensor(
            cont_sh.data(), 4, ascend::toAclDtype(x.dtype()),
            cont_st.data(), 0, ACL_FORMAT_ND, cont_storage.data(), 1,
            x_rot_dev);
        aclTensor* t_out_rot = aclCreateTensor(
            cont_sh.data(), 4, ascend::toAclDtype(x.dtype()),
            cont_st.data(), 0, ACL_FORMAT_ND, cont_storage.data(), 1,
            out_rot_dev);

        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        aclnnRotaryPositionEmbeddingGetWorkspaceSize(
            t_x_rot, t_cos, t_sin, mode, t_out_rot, &ws, &exec);
        auto& arena = ascend::workspacePool().ensure(stream, ws);
        aclnnRotaryPositionEmbedding(arena.buf, ws, exec, stream);

        // Scatter rotated part to output.
        for (int64_t i = 0; i < T * N; ++i) {
          aclrtMemcpyAsync(
              static_cast<uint8_t*>(const_cast<void*>(out.data())) +
                  i * D * elem_sz,
              static_cast<size_t>(R) * elem_sz,
              static_cast<uint8_t*>(out_rot_dev) + i * R * elem_sz,
              static_cast<size_t>(R) * elem_sz,
              ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        }

        // Copy non-rotary part x[..., R:] -> out[..., R:].
        int64_t pass = D - R;
        for (int64_t i = 0; i < T * N; ++i) {
          aclrtMemcpyAsync(
              static_cast<uint8_t*>(const_cast<void*>(out.data())) +
                  (i * D + R) * elem_sz,
              static_cast<size_t>(pass) * elem_sz,
              static_cast<const uint8_t*>(x.data()) + (i * D + R) * elem_sz,
              static_cast<size_t>(pass) * elem_sz,
              ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        }

        aclDestroyTensor(t_x_rot);
        aclDestroyTensor(t_out_rot);
        aclrtFree(x_rot_dev);
        aclrtFree(out_rot_dev);
      } else {
        // Full: R == D.
        std::vector<int64_t> full_sh = {1, T, N, D};
        std::vector<int64_t> full_st = {T * N * D, N * D, D, 1};
        aclTensor* t_x = detail::reshapeAclView(x, full_sh, full_st);
        aclTensor* t_out = detail::reshapeAclView(out, full_sh, full_st);

        uint64_t ws = 0;
        aclOpExecutor* exec = nullptr;
        aclnnRotaryPositionEmbeddingGetWorkspaceSize(
            t_x, t_cos, t_sin, mode, t_out, &ws, &exec);
        auto& arena = ascend::workspacePool().ensure(stream, ws);
        aclnnRotaryPositionEmbedding(arena.buf, ws, exec, stream);

        aclDestroyTensor(t_x);
        aclDestroyTensor(t_out);
      }
    };

    apply_rope(query, query_out, Nq);
    apply_rope(key, key_out, Nkv);

    aclDestroyTensor(t_cos);
    aclDestroyTensor(t_sin);
    aclrtFree(cos_dev);
    aclrtFree(sin_dev);
  }
};

}  // namespace infini::ops

#endif
