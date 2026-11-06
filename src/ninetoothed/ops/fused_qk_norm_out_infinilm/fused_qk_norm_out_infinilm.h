#ifndef INFINI_OPS_NINETOOTHED_FUSED_QK_NORM_OUT_INFINILM_H_
#define INFINI_OPS_NINETOOTHED_FUSED_QK_NORM_OUT_INFINILM_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "base/fused_qk_norm_out_infinilm.h"
#include "data_type.h"
#include "fused_qk_norm_out_infinilm/infini_ops_ninetoothed_fused_qk_norm_out_infinilm.h"

namespace infini::ops {

template <>
class Operator<FusedQkNormOutInfinilm, Device::Type::kNvidia, 9>
    : public FusedQkNormOutInfinilm {
 public:
  using FusedQkNormOutInfinilm::FusedQkNormOutInfinilm;
  using FusedQkNormOutInfinilm::operator();

  static bool supports(Tensor qkv, int64_t num_heads_q, int64_t num_heads_k,
                       int64_t num_heads_v, int64_t head_dim, float,
                       Tensor q_weight, Tensor k_weight, Tensor q_output,
                       Tensor k_output) {
    return qkv.ndim() == 2 && qkv.IsContiguous() && q_output.ndim() == 2 &&
           q_output.IsContiguous() && k_output.ndim() == 2 &&
           k_output.IsContiguous() && q_weight.ndim() == 1 &&
           q_weight.stride(0) == 1 && k_weight.ndim() == 1 &&
           k_weight.stride(0) == 1 && num_heads_q == 16 && num_heads_k == 8 &&
           num_heads_v == 8 && head_dim == 128 &&
           qkv.dtype() == DataType::kBFloat16 &&
           q_weight.dtype() == qkv.dtype() && k_weight.dtype() == qkv.dtype() &&
           q_output.dtype() == qkv.dtype() && k_output.dtype() == qkv.dtype();
  }

  void operator()(Tensor qkv, int64_t num_heads_q, int64_t num_heads_k,
                  int64_t num_heads_v, int64_t head_dim, float eps,
                  Tensor q_weight, Tensor k_weight, Tensor q_output,
                  Tensor k_output) const override {
    const auto tokens = static_cast<std::uint64_t>(qkv.size(0));
    const auto total_width = static_cast<std::int64_t>(qkv.size(1));
    const auto q_width = static_cast<std::int64_t>(num_heads_q * head_dim);
    const auto k_width = static_cast<std::int64_t>(num_heads_k * head_dim);
    const auto block_width = static_cast<std::int64_t>(k_width);
    const auto element_size = qkv.element_size();

    std::vector<std::uint64_t> block_sizes{tokens, 8, 128};
    std::vector<std::int64_t> input_strides{total_width, head_dim, 1};
    std::vector<std::int64_t> q_output_strides{q_width, head_dim, 1};
    std::vector<std::int64_t> k_output_strides{k_width, head_dim, 1};
    std::vector<std::int64_t> weight_strides{0, 0, 1};
    std::uint64_t scalar_shape[1] = {};
    std::int64_t scalar_strides[1] = {};
    double eps_value = static_cast<double>(eps);
    std::int64_t width_value = static_cast<std::int64_t>(head_dim);

    auto* input_data = static_cast<std::byte*>(qkv.data());
    auto* query_output_data = static_cast<std::byte*>(q_output.data());
    auto* q0_input_data = input_data;
    auto* q1_input_data = input_data + block_width * element_size;
    auto* k_input_data = input_data + q_width * element_size;
    auto* q0_output_data = query_output_data;
    auto* q1_output_data = query_output_data + block_width * element_size;

    ::NineToothedTensor q0_input{q0_input_data, block_sizes.data(),
                                 input_strides.data()};
    ::NineToothedTensor q1_input{q1_input_data, block_sizes.data(),
                                 input_strides.data()};
    ::NineToothedTensor k_input{k_input_data, block_sizes.data(),
                                input_strides.data()};
    ::NineToothedTensor q0_weight{q_weight.data(), block_sizes.data(),
                                  weight_strides.data()};
    ::NineToothedTensor q1_weight{q_weight.data(), block_sizes.data(),
                                  weight_strides.data()};
    ::NineToothedTensor k_weight_tensor{k_weight.data(), block_sizes.data(),
                                        weight_strides.data()};
    ::NineToothedTensor eps_tensor{&eps_value, scalar_shape, scalar_strides};
    ::NineToothedTensor q0_output{q0_output_data, block_sizes.data(),
                                  q_output_strides.data()};
    ::NineToothedTensor q1_output{q1_output_data, block_sizes.data(),
                                  q_output_strides.data()};
    ::NineToothedTensor k_output_tensor{k_output.data(), block_sizes.data(),
                                        k_output_strides.data()};
    ::NineToothedTensor width_tensor{&width_value, scalar_shape,
                                     scalar_strides};

    const auto result =
        launch_infini_ops_ninetoothed_fused_qk_norm_out_infinilm(
            static_cast<NineToothedStream>(stream_), q0_input, q1_input,
            k_input, q0_weight, q1_weight, k_weight_tensor, eps_tensor,
            q0_output, q1_output, k_output_tensor, width_tensor, 9);
    assert(result == 0 && "NineToothed `FusedQkNormOutInfinilm` launch failed");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NINETOOTHED_FUSED_QK_NORM_OUT_INFINILM_H_
