#ifndef INFINI_OPS_ASCEND_LIGHTNING_ATTENTION_INFINILM_KERNEL_H_
#define INFINI_OPS_ASCEND_LIGHTNING_ATTENTION_INFINILM_KERNEL_H_

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_add.h"
#include "aclnn_mul.h"
#include "aclnnop/aclnn_matmul.h"
#include "base/lightning_attention_infinilm.h"
#include "data_type.h"
#include "native/ascend/common.h"
#include "native/ascend/data_type_.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

// Ascend implementation of the indexed-pool lightning attention.
//
// CANN does not provide a fused lightning-attention operator, so this path
// composes aclnn kernels and walks requests and tokens on the host:
//
//   ratio   = exp(-slope)                           (per head, once per call)
//   decayed = Mul(state, ratio)                     (ratio is [H, 1, 1])
//   outer   = Matmul(k_t [H, D, 1], v_t [H, 1, D])
//   state   = Add(decayed, outer)
//   out_t   = Matmul(q_t [H, 1, D], state)
//
// Each request is staged in a workspace row.  The source state row is copied
// into the workspace first, all tokens are evaluated there, and the result is
// copied back to the destination row last.  Therefore the source row is never
// written, even when the source and destination rows are different.
//
// This is a correctness-first fallback.  A fused AscendC kernel is the
// follow-up once the composition is validated on hardware.
template <>
class Operator<LightningAttentionInfinilm, Device::Type::kAscend>
    : public LightningAttentionInfinilm {
 public:
  Operator(const Tensor q, const Tensor k, const Tensor v, const Tensor slope,
           Tensor initial_state, const Tensor initial_state_indices,
           const Tensor final_state_indices, Tensor out)
      : LightningAttentionInfinilm(q, k, v, slope, initial_state,
                                   initial_state_indices, final_state_indices,
                                   out),
        state_cache_({static_cast<int64_t>(num_heads_),
                      static_cast<int64_t>(head_dim_),
                      static_cast<int64_t>(head_dim_)},
                     ascend::ToAclDtype(data_type_), nullptr),
        decayed_cache_({static_cast<int64_t>(num_heads_),
                        static_cast<int64_t>(head_dim_),
                        static_cast<int64_t>(head_dim_)},
                       ascend::ToAclDtype(data_type_), nullptr),
        outer_cache_({static_cast<int64_t>(num_heads_),
                      static_cast<int64_t>(head_dim_),
                      static_cast<int64_t>(head_dim_)},
                     ascend::ToAclDtype(data_type_), nullptr),
        ratio_cache_({static_cast<int64_t>(num_heads_), 1, 1},
                     ascend::ToAclDtype(data_type_), nullptr),
        outer_k_cache_({static_cast<int64_t>(num_heads_),
                        static_cast<int64_t>(head_dim_), 1},
                       ascend::ToAclDtype(data_type_), nullptr),
        outer_v_cache_({static_cast<int64_t>(num_heads_), 1,
                        static_cast<int64_t>(head_dim_)},
                       ascend::ToAclDtype(data_type_), nullptr),
        output_q_cache_({static_cast<int64_t>(num_heads_), 1,
                         static_cast<int64_t>(head_dim_)},
                        ascend::ToAclDtype(data_type_), nullptr),
        output_cache_({static_cast<int64_t>(num_heads_), 1,
                       static_cast<int64_t>(head_dim_)},
                      ascend::ToAclDtype(data_type_), nullptr) {
    assert((data_type_ == DataType::kFloat16 ||
            data_type_ == DataType::kBFloat16 ||
            data_type_ == DataType::kFloat32) &&
           "`LightningAttentionInfinilm` Ascend path requires float16, "
           "bfloat16 or float32");
    assert(q.IsContiguous() && k.IsContiguous() && v.IsContiguous() &&
           out.IsContiguous() && initial_state.IsContiguous() &&
           "`LightningAttentionInfinilm` Ascend path requires contiguous "
           "tensors");
    assert(num_heads_ > 0 && head_dim_ > 0 &&
           "`LightningAttentionInfinilm` Ascend path requires non-empty "
           "heads and head_dim");

    add_alpha_storage_ = 1.0f;
    add_alpha_ = aclCreateScalar(&add_alpha_storage_, ACL_FLOAT);
    assert(add_alpha_ != nullptr &&
           "`LightningAttentionInfinilm` failed to create the Add alpha "
           "scalar");
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors: they are still referenced by the Repeatable
    // executors.  See `AclTensorCache::release()`.
    state_cache_.release();
    decayed_cache_.release();
    outer_cache_.release();
    ratio_cache_.release();
    outer_k_cache_.release();
    outer_v_cache_.release();
    output_q_cache_.release();
    output_cache_.release();

    if (add_alpha_) aclDestroyScalar(add_alpha_);
  }

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor slope, Tensor initial_state,
                  const Tensor initial_state_indices,
                  const Tensor final_state_indices, Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    const size_t element_size = kDataTypeToSize.at(data_type_);
    const size_t head_elements = head_dim_ * head_dim_;
    const size_t state_bytes = num_heads_ * head_elements * element_size;
    const size_t ratio_bytes = num_heads_ * element_size;

    // The host reads slope and state indices, and queues device copies below.
    // Synchronize once so both reads observe all earlier work on this stream.
    CheckAcl(aclrtSynchronizeStream(stream),
             "aclrtSynchronizeStream(lightning attention input)");

    auto& pool = ascend::GetWorkspacePool();
    auto& state_arena =
        pool.Ensure(stream, state_bytes, "lightning_attention_state");
    auto& decayed_arena =
        pool.Ensure(stream, state_bytes, "lightning_attention_decayed");
    auto& outer_arena =
        pool.Ensure(stream, state_bytes, "lightning_attention_outer");
    auto& ratio_arena =
        pool.Ensure(stream, ratio_bytes, "lightning_attention_ratio");

    BuildRatio(slope, ratio_arena.buf, element_size);

    std::vector<int64_t> initial_rows(batch_size_);
    std::vector<int64_t> final_rows(batch_size_);
    ReadIndices(initial_state_indices, initial_rows);
    ReadIndices(final_state_indices, final_rows);

    void* state_buf = state_arena.buf;
    void* decayed_buf = decayed_arena.buf;
    void* outer_buf = outer_arena.buf;
    const void* ratio_buf = ratio_arena.buf;

    for (Tensor::Size b = 0; b < batch_size_; ++b) {
      const auto* initial_row_ptr =
          static_cast<const char*>(initial_state.data()) +
          initial_rows[b] * state_pool_stride_ * element_size;
      auto* final_row_ptr =
          static_cast<char*>(initial_state.data()) +
          final_rows[b] * state_pool_stride_ * element_size;

      // Stage the source row in the workspace.  The destination row is not
      // touched until the final writeback, so an aliased source row remains
      // read-only for the whole request.
      CheckAcl(aclrtMemcpyAsync(state_buf, state_bytes, initial_row_ptr,
                                state_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                stream),
               "aclrtMemcpyAsync(state staging)");

      for (Tensor::Size t = 0; t < seq_len_; ++t) {
        const auto* q_t =
            static_cast<const char*>(q.data()) +
            (b * q_batch_stride_ + t * q_seq_stride_) * element_size;
        const auto* k_t =
            static_cast<const char*>(k.data()) +
            (b * k_batch_stride_ + t * k_seq_stride_) * element_size;
        const auto* v_t =
            static_cast<const char*>(v.data()) +
            (b * v_batch_stride_ + t * v_seq_stride_) * element_size;
        auto* out_t =
            static_cast<char*>(out.data()) +
            (b * out_batch_stride_ + t * out_seq_stride_) * element_size;

        MulDecay(state_buf, ratio_buf, decayed_buf, stream);
        OuterProduct(k_t, v_t, outer_buf, stream);
        AddState(decayed_buf, outer_buf, state_buf, stream);
        WriteOutput(q_t, state_buf, out_t, stream);
      }

      CheckAcl(aclrtMemcpyAsync(final_row_ptr, state_bytes, state_buf,
                                state_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                stream),
               "aclrtMemcpyAsync(state writeback)");
    }
  }

 private:
  void CheckAcl(aclError error, const char* what) const {
    (void)what;
    assert(error == ACL_SUCCESS);
  }

  // Materialize exp(-slope) once per forward call.  The ratio buffer is
  // [heads, 1, 1] so that aclnnMul broadcasts it over the [heads, head_dim,
  // head_dim] state.
  void BuildRatio(const Tensor& slope, void* ratio_buf,
                  size_t element_size) const {
    std::vector<float> host_slope(num_heads_);
    CheckAcl(aclrtMemcpy(host_slope.data(), num_heads_ * sizeof(float),
                         const_cast<void*>(slope.data()),
                         num_heads_ * sizeof(float),
                         ACL_MEMCPY_DEVICE_TO_HOST),
             "aclrtMemcpy(slope to host)");

    std::vector<uint8_t> host_ratio(num_heads_ * element_size);
    for (Tensor::Size h = 0; h < num_heads_; ++h) {
      const float ratio = std::exp(-host_slope[h]);
      auto* destination = host_ratio.data() + h * element_size;

      if (data_type_ == DataType::kFloat32) {
        std::memcpy(destination, &ratio, sizeof(float));
      } else if (data_type_ == DataType::kFloat16) {
        const auto converted = Float16::FromFloat(ratio);
        std::memcpy(destination, &converted, sizeof(converted));
      } else {
        const auto converted = BFloat16::FromFloat(ratio);
        std::memcpy(destination, &converted, sizeof(converted));
      }
    }

    CheckAcl(aclrtMemcpy(ratio_buf, num_heads_ * element_size,
                         host_ratio.data(), num_heads_ * element_size,
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(ratio to device)");
  }

  void ReadIndices(const Tensor& indices, std::vector<int64_t>& host) const {
    host.resize(batch_size_);
    if (index_dtype_ == DataType::kInt64) {
      CheckAcl(aclrtMemcpy(host.data(), batch_size_ * sizeof(int64_t),
                           const_cast<void*>(indices.data()),
                           batch_size_ * sizeof(int64_t),
                           ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy(indices int64)");
    } else {
      std::vector<int32_t> narrowed(batch_size_);
      CheckAcl(aclrtMemcpy(narrowed.data(), batch_size_ * sizeof(int32_t),
                           const_cast<void*>(indices.data()),
                           batch_size_ * sizeof(int32_t),
                           ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy(indices int32)");
      for (Tensor::Size b = 0; b < batch_size_; ++b) {
        host[b] = static_cast<int64_t>(narrowed[b]);
      }
    }
  }

  void MulDecay(const void* state, const void* ratio, void* decayed,
                aclrtStream stream) const {
    auto* t_state = state_cache_.get(const_cast<void*>(state));
    auto* t_ratio = ratio_cache_.get(const_cast<void*>(ratio));
    auto* t_decayed = decayed_cache_.get(decayed);

    if (!mul_exec_) {
      CheckAcl(aclnnMulGetWorkspaceSize(t_state, t_ratio, t_decayed, &mul_ws_,
                                        &mul_exec_),
               "aclnnMulGetWorkspaceSize");
      aclSetAclOpExecutorRepeatable(mul_exec_);
    } else {
      aclSetInputTensorAddr(mul_exec_, 0, t_state, const_cast<void*>(state));
      aclSetInputTensorAddr(mul_exec_, 1, t_ratio, const_cast<void*>(ratio));
      aclSetOutputTensorAddr(mul_exec_, 0, t_decayed, decayed);
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, mul_ws_, "lightning_attention_mul_workspace");
    CheckAcl(aclnnMul(arena.buf, mul_ws_, mul_exec_, stream), "aclnnMul");
  }

  void OuterProduct(const void* k_t, const void* v_t, void* outer,
                    aclrtStream stream) const {
    auto* t_k = outer_k_cache_.get(const_cast<void*>(k_t));
    auto* t_v = outer_v_cache_.get(const_cast<void*>(v_t));
    auto* t_outer = outer_cache_.get(outer);

    if (!outer_exec_) {
      const int8_t cube_math_type = data_type_ == DataType::kFloat32 ? 0 : 1;
      CheckAcl(aclnnMatmulGetWorkspaceSize(t_k, t_v, t_outer, cube_math_type,
                                           &outer_ws_, &outer_exec_),
               "aclnnMatmulGetWorkspaceSize(outer)");
      aclSetAclOpExecutorRepeatable(outer_exec_);
    } else {
      aclSetInputTensorAddr(outer_exec_, 0, t_k, const_cast<void*>(k_t));
      aclSetInputTensorAddr(outer_exec_, 1, t_v, const_cast<void*>(v_t));
      aclSetOutputTensorAddr(outer_exec_, 0, t_outer, outer);
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, outer_ws_, "lightning_attention_outer_workspace");
    CheckAcl(aclnnMatmul(arena.buf, outer_ws_, outer_exec_, stream),
             "aclnnMatmul(outer)");
  }

  void AddState(const void* decayed, const void* outer, void* state,
                aclrtStream stream) const {
    auto* t_decayed = decayed_cache_.get(const_cast<void*>(decayed));
    auto* t_outer = outer_cache_.get(const_cast<void*>(outer));
    auto* t_state = state_cache_.get(state);

    if (!add_exec_) {
      CheckAcl(aclnnAddGetWorkspaceSize(t_decayed, t_outer, add_alpha_, t_state,
                                        &add_ws_, &add_exec_),
               "aclnnAddGetWorkspaceSize");
      aclSetAclOpExecutorRepeatable(add_exec_);
    } else {
      aclSetInputTensorAddr(add_exec_, 0, t_decayed,
                            const_cast<void*>(decayed));
      aclSetInputTensorAddr(add_exec_, 1, t_outer, const_cast<void*>(outer));
      aclSetOutputTensorAddr(add_exec_, 0, t_state, state);
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, add_ws_, "lightning_attention_add_workspace");
    CheckAcl(aclnnAdd(arena.buf, add_ws_, add_exec_, stream), "aclnnAdd");
  }

  void WriteOutput(const void* q_t, const void* state, void* out_t,
                   aclrtStream stream) const {
    auto* t_q = output_q_cache_.get(const_cast<void*>(q_t));
    auto* t_state = state_cache_.get(const_cast<void*>(state));
    auto* t_out = output_cache_.get(out_t);

    if (!output_exec_) {
      const int8_t cube_math_type = data_type_ == DataType::kFloat32 ? 0 : 1;
      CheckAcl(aclnnMatmulGetWorkspaceSize(t_q, t_state, t_out, cube_math_type,
                                           &output_ws_, &output_exec_),
               "aclnnMatmulGetWorkspaceSize(output)");
      aclSetAclOpExecutorRepeatable(output_exec_);
    } else {
      aclSetInputTensorAddr(output_exec_, 0, t_q, const_cast<void*>(q_t));
      aclSetInputTensorAddr(output_exec_, 1, t_state, const_cast<void*>(state));
      aclSetOutputTensorAddr(output_exec_, 0, t_out, out_t);
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(
        stream, output_ws_, "lightning_attention_output_workspace");
    CheckAcl(aclnnMatmul(arena.buf, output_ws_, output_exec_, stream),
             "aclnnMatmul(output)");
  }

  mutable ascend::AclTensorCache state_cache_;

  mutable ascend::AclTensorCache decayed_cache_;

  mutable ascend::AclTensorCache outer_cache_;

  mutable ascend::AclTensorCache ratio_cache_;

  mutable ascend::AclTensorCache outer_k_cache_;

  mutable ascend::AclTensorCache outer_v_cache_;

  mutable ascend::AclTensorCache output_q_cache_;

  mutable ascend::AclTensorCache output_cache_;

  float add_alpha_storage_ = 1.0f;

  aclScalar* add_alpha_ = nullptr;

  mutable aclOpExecutor* mul_exec_ = nullptr;

  mutable uint64_t mul_ws_ = 0;

  mutable aclOpExecutor* outer_exec_ = nullptr;

  mutable uint64_t outer_ws_ = 0;

  mutable aclOpExecutor* add_exec_ = nullptr;

  mutable uint64_t add_ws_ = 0;

  mutable aclOpExecutor* output_exec_ = nullptr;

  mutable uint64_t output_ws_ = 0;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_LIGHTNING_ATTENTION_INFINILM_KERNEL_H_