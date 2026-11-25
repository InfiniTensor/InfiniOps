#ifndef INFINI_OPS_CPU_LIGHTNING_ATTENTION_INFINILM_H_
#define INFINI_OPS_CPU_LIGHTNING_ATTENTION_INFINILM_H_

#include <cmath>
#include <cstdint>
#include <vector>

#include "base/lightning_attention_infinilm.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "native/cpu/caster_.h"
#include "tensor.h"

namespace infini::ops {

template <>
class Operator<LightningAttentionInfinilm, Device::Type::kCpu>
    : public LightningAttentionInfinilm, Caster<Device::Type::kCpu> {
 public:
  Operator(const Tensor q, const Tensor k, const Tensor v, const Tensor slope,
           Tensor initial_state, const Tensor initial_state_indices,
           const Tensor final_state_indices, Tensor out)
      : LightningAttentionInfinilm{q,     k,     v,     slope, initial_state,
                                   initial_state_indices, final_state_indices,
                                   out} {}

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor slope, Tensor initial_state,
                  const Tensor initial_state_indices,
                  const Tensor final_state_indices, Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        out.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(q, k, v, slope, initial_state, initial_state_indices,
                     final_state_indices, out);
        },
        "`Operator<LightningAttentionInfinilm, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(const Tensor q, const Tensor k, const Tensor v,
               const Tensor slope, Tensor initial_state,
               const Tensor initial_state_indices,
               const Tensor final_state_indices, Tensor out) const {
    const auto* q_ptr = static_cast<const T*>(q.data());
    const auto* k_ptr = static_cast<const T*>(k.data());
    const auto* v_ptr = static_cast<const T*>(v.data());
    const auto* slope_ptr = static_cast<const float*>(slope.data());
    auto* state_ptr = static_cast<T*>(initial_state.data());
    auto* out_ptr = static_cast<T*>(out.data());

    const bool int64_indices = index_dtype_ == DataType::kInt64;
    const auto* initial_indices = initial_state_indices.data();
    const auto* final_indices = final_state_indices.data();

    // The recurrent state of one request, accumulated in float32.
    std::vector<float> state(num_heads_ * head_dim_ * head_dim_);

    for (Tensor::Size b = 0; b < batch_size_; ++b) {
      Tensor::Size initial_row;
      Tensor::Size final_row;
      if (int64_indices) {
        initial_row = static_cast<Tensor::Size>(
            static_cast<const int64_t*>(initial_indices)[b]);
        final_row =
            static_cast<Tensor::Size>(static_cast<const int64_t*>(final_indices)[b]);
      } else {
        initial_row = static_cast<Tensor::Size>(
            static_cast<const int32_t*>(initial_indices)[b]);
        final_row =
            static_cast<Tensor::Size>(static_cast<const int32_t*>(final_indices)[b]);
      }

      const T* initial_row_ptr = state_ptr + initial_row * state_pool_stride_;
      for (Tensor::Size h = 0; h < num_heads_; ++h) {
        const T* head_ptr = initial_row_ptr + h * state_head_stride_;
        float* state_head = state.data() + h * head_dim_ * head_dim_;
        for (Tensor::Size i = 0; i < head_dim_; ++i) {
          for (Tensor::Size j = 0; j < head_dim_; ++j) {
            state_head[i * head_dim_ + j] =
                Cast<float>(head_ptr[i * state_row_stride_ +
                                     j * state_column_stride_]);
          }
        }
      }

      for (Tensor::Size t = 0; t < seq_len_; ++t) {
        for (Tensor::Size h = 0; h < num_heads_; ++h) {
          const float ratio = std::exp(-slope_ptr[h * slope_stride_]);
          const T* q_row = q_ptr + b * q_batch_stride_ + t * q_seq_stride_ +
                           h * q_head_stride_;
          const T* k_row = k_ptr + b * k_batch_stride_ + t * k_seq_stride_ +
                           h * k_head_stride_;
          const T* v_row = v_ptr + b * v_batch_stride_ + t * v_seq_stride_ +
                           h * v_head_stride_;
          float* state_head = state.data() + h * head_dim_ * head_dim_;

          for (Tensor::Size i = 0; i < head_dim_; ++i) {
            const float k_i = Cast<float>(k_row[i]);
            for (Tensor::Size j = 0; j < head_dim_; ++j) {
              state_head[i * head_dim_ + j] =
                  ratio * state_head[i * head_dim_ + j] +
                  k_i * Cast<float>(v_row[j]);
            }
          }

          T* out_row = out_ptr + b * out_batch_stride_ + t * out_seq_stride_ +
                       h * out_head_stride_;
          for (Tensor::Size j = 0; j < head_dim_; ++j) {
            float acc = 0.0f;
            for (Tensor::Size i = 0; i < head_dim_; ++i) {
              acc += Cast<float>(q_row[i]) * state_head[i * head_dim_ + j];
            }
            out_row[j] = Cast<T>(acc);
          }
        }
      }

      // Only the destination row is updated; the initial row stays untouched.
      T* final_row_ptr = state_ptr + final_row * state_pool_stride_;
      for (Tensor::Size h = 0; h < num_heads_; ++h) {
        T* head_ptr = final_row_ptr + h * state_head_stride_;
        const float* state_head = state.data() + h * head_dim_ * head_dim_;
        for (Tensor::Size i = 0; i < head_dim_; ++i) {
          for (Tensor::Size j = 0; j < head_dim_; ++j) {
            head_ptr[i * state_row_stride_ + j * state_column_stride_] =
                Cast<T>(state_head[i * head_dim_ + j]);
          }
        }
      }
    }
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CPU_LIGHTNING_ATTENTION_INFINILM_H_
