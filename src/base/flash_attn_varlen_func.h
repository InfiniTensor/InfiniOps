#ifndef INFINI_OPS_BASE_FLASH_ATTN_VARLEN_FUNC_H_
#define INFINI_OPS_BASE_FLASH_ATTN_VARLEN_FUNC_H_

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "operator.h"

namespace infini::ops {

// Packed variable-length attention aligned with Dao-AILab FlashAttention's
// `flash_attn_varlen_func` public interface.
class FlashAttnVarlenFunc : public Operator<FlashAttnVarlenFunc> {
 public:
  FlashAttnVarlenFunc(const Tensor q, const Tensor k, const Tensor v,
                      const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                      const int64_t max_seqlen_q, const int64_t max_seqlen_k,
                      Tensor out)
      : FlashAttnVarlenFunc{q,
                            k,
                            v,
                            cu_seqlens_q,
                            cu_seqlens_k,
                            max_seqlen_q,
                            max_seqlen_k,
                            0.0,
                            std::nullopt,
                            false,
                            {-1, -1},
                            0.0,
                            std::nullopt,
                            false,
                            false,
                            std::nullopt,
                            out} {}

  FlashAttnVarlenFunc(const Tensor q, const Tensor k, const Tensor v,
                      const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                      const int64_t max_seqlen_q, const int64_t max_seqlen_k,
                      const double dropout_p,
                      const std::optional<double> softmax_scale,
                      const bool causal, const std::vector<int64_t> window_size,
                      const double softcap,
                      const std::optional<Tensor> alibi_slopes,
                      const bool deterministic, const bool return_attn_probs,
                      const std::optional<Tensor> block_table, Tensor out)
      : q_shape_{q.shape()},
        k_shape_{k.shape()},
        v_shape_{v.shape()},
        cu_seqlens_q_shape_{cu_seqlens_q.shape()},
        cu_seqlens_k_shape_{cu_seqlens_k.shape()},
        out_shape_{out.shape()},
        q_strides_{q.strides()},
        k_strides_{k.strides()},
        v_strides_{v.strides()},
        cu_seqlens_q_strides_{cu_seqlens_q.strides()},
        cu_seqlens_k_strides_{cu_seqlens_k.strides()},
        out_strides_{out.strides()},
        q_dtype_{q.dtype()},
        k_dtype_{k.dtype()},
        v_dtype_{v.dtype()},
        cu_seqlens_q_dtype_{cu_seqlens_q.dtype()},
        cu_seqlens_k_dtype_{cu_seqlens_k.dtype()},
        out_dtype_{out.dtype()},
        device_index_{q.device().index()} {
    assert(q.ndim() == 3 && k.ndim() == 3 && v.ndim() == 3 &&
           "`FlashAttnVarlenFunc` requires packed 3D Q, K, and V tensors");
    assert(k.shape() == v.shape() &&
           "`FlashAttnVarlenFunc` requires K and V to have the same shape");
    assert(q.size(1) > 0 && k.size(1) > 0 && q.size(2) == k.size(2) &&
           q.size(1) % k.size(1) == 0 &&
           "`FlashAttnVarlenFunc` requires compatible Q and KV heads");
    assert(q.size(2) > 0 && q.size(2) <= 256 && q.size(2) % 8 == 0 &&
           "`FlashAttnVarlenFunc` requires a head dimension divisible by 8 "
           "and no greater than 256");
    assert(out.shape() == q.shape() &&
           "`FlashAttnVarlenFunc` output must have the same shape as Q");
    assert(
        (q_dtype_ == DataType::kFloat16 || q_dtype_ == DataType::kBFloat16) &&
        q_dtype_ == k_dtype_ && q_dtype_ == v_dtype_ &&
        q_dtype_ == out_dtype_ &&
        "`FlashAttnVarlenFunc` requires matching float16 or bfloat16 Q, "
        "K, V, and output dtypes");
    assert(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1 &&
           out.stride(-1) == 1 &&
           "`FlashAttnVarlenFunc` requires contiguous head dimensions");
    assert(cu_seqlens_q.ndim() == 1 && cu_seqlens_k.ndim() == 1 &&
           cu_seqlens_q.shape() == cu_seqlens_k.shape() &&
           cu_seqlens_q.numel() >= 2 &&
           "`FlashAttnVarlenFunc` cumulative sequence tensors must be "
           "matching non-empty vectors");
    assert(cu_seqlens_q_dtype_ == DataType::kInt32 &&
           cu_seqlens_k_dtype_ == DataType::kInt32 &&
           cu_seqlens_q.IsContiguous() && cu_seqlens_k.IsContiguous() &&
           "`FlashAttnVarlenFunc` cumulative sequence tensors must be "
           "contiguous int32 tensors");
    assert(max_seqlen_q > 0 && max_seqlen_k > 0 &&
           "`FlashAttnVarlenFunc` maximum sequence lengths must be positive");
    assert(window_size.size() == 2 && window_size[0] >= -1 &&
           window_size[1] >= -1 &&
           "`FlashAttnVarlenFunc` `window_size` must contain two values >= -1");

    assert(dropout_p == 0.0 &&
           "`FlashAttnVarlenFunc` initially supports inference only");
    assert(softcap == 0.0 &&
           "`FlashAttnVarlenFunc` does not yet support softcap");
    assert(!deterministic &&
           "`FlashAttnVarlenFunc` does not yet support deterministic mode");
    assert(!return_attn_probs &&
           "`FlashAttnVarlenFunc` does not yet return attention probabilities");
    assert(!block_table.has_value() &&
           "`FlashAttnVarlenFunc` does not yet support paged KV cache");
    assert(!alibi_slopes.has_value() &&
           "`FlashAttnVarlenFunc` does not yet support ALiBi slopes");

    const auto same_device_as_q = [&](const Tensor tensor) {
      return tensor.device().type() == q.device().type() &&
             tensor.device().index() == q.device().index();
    };
    assert(same_device_as_q(k) && same_device_as_q(v) &&
           same_device_as_q(cu_seqlens_q) && same_device_as_q(cu_seqlens_k) &&
           same_device_as_q(out) &&
           "`FlashAttnVarlenFunc` tensors must be on the same device");

    (void)softmax_scale;
    (void)causal;
  }

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor cu_seqlens_q, const Tensor cu_seqlens_k,
                  const int64_t max_seqlen_q, const int64_t max_seqlen_k,
                  Tensor out) const {
    (*this)(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            0.0, std::nullopt, false, {-1, -1}, 0.0, std::nullopt, false, false,
            std::nullopt, out);
  }

  virtual void operator()(
      const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
      const Tensor cu_seqlens_k, const int64_t max_seqlen_q,
      const int64_t max_seqlen_k, const double dropout_p,
      const std::optional<double> softmax_scale, const bool causal,
      const std::vector<int64_t> window_size, const double softcap,
      const std::optional<Tensor> alibi_slopes, const bool deterministic,
      const bool return_attn_probs, const std::optional<Tensor> block_table,
      Tensor out) const = 0;

 protected:
  Tensor::Shape q_shape_;

  Tensor::Shape k_shape_;

  Tensor::Shape v_shape_;

  Tensor::Shape cu_seqlens_q_shape_;

  Tensor::Shape cu_seqlens_k_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides q_strides_;

  Tensor::Strides k_strides_;

  Tensor::Strides v_strides_;

  Tensor::Strides cu_seqlens_q_strides_;

  Tensor::Strides cu_seqlens_k_strides_;

  Tensor::Strides out_strides_;

  DataType q_dtype_;

  DataType k_dtype_;

  DataType v_dtype_;

  DataType cu_seqlens_q_dtype_;

  DataType cu_seqlens_k_dtype_;

  DataType out_dtype_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_FLASH_ATTN_VARLEN_FUNC_H_
