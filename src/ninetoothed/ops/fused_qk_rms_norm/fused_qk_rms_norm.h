#ifndef INFINI_OPS_NINETOOTHED_FUSED_QK_RMS_NORM_H_
#define INFINI_OPS_NINETOOTHED_FUSED_QK_RMS_NORM_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "base/fused_qk_rms_norm.h"
#include "data_type.h"
#include "fused_qk_rms_norm/infini_ops_ninetoothed_fused_qk_rms_norm.h"

namespace infini::ops {

template <>
class Operator<FusedQkRmsNorm, Device::Type::kNvidia, 9>
    : public FusedQkRmsNorm {
 public:
  using FusedQkRmsNorm::FusedQkRmsNorm;
  using FusedQkRmsNorm::operator();

  static bool supports(Tensor q, Tensor kv, float, Tensor q_weight,
                       Tensor kv_weight, Tensor q_output, Tensor kv_output) {
    return q.ndim() == 3 && kv.ndim() == 3 && q.size(1) == 16 &&
           kv.size(1) == 8 && q.size(2) == 128 && kv.size(2) == 128 &&
           q.stride(1) == 128 && kv.stride(1) == 128 && q.stride(2) == 1 &&
           kv.stride(2) == 1 && q_weight.ndim() == 1 &&
           q_weight.size(0) == 128 && q_weight.stride(0) == 1 &&
           kv_weight.ndim() == 1 && kv_weight.size(0) == 128 &&
           kv_weight.stride(0) == 1 && q_output.shape() == q.shape() &&
           q_output.IsContiguous() && kv_output.shape() == kv.shape() &&
           kv_output.IsContiguous() && q.dtype() == DataType::kBFloat16 &&
           q_weight.dtype() == q.dtype() && kv_weight.dtype() == q.dtype() &&
           q_output.dtype() == q.dtype() && kv_output.dtype() == q.dtype();
  }

  void operator()(Tensor q, Tensor kv, Tensor q_weight, Tensor kv_weight,
                  float eps, Tensor q_output, Tensor kv_output) const override {
    const auto tokens = static_cast<std::uint64_t>(q.size(0));
    const auto block_width = static_cast<std::int64_t>(kv.size(1) * kv.size(2));
    const auto element_size = q.element_size();

    std::vector<std::uint64_t> block_sizes{tokens, 8, 128};
    std::vector<std::int64_t> q_input_strides{
        static_cast<std::int64_t>(q.stride(0)),
        static_cast<std::int64_t>(q.stride(1)),
        static_cast<std::int64_t>(q.stride(2))};
    std::vector<std::int64_t> kv_input_strides{
        static_cast<std::int64_t>(kv.stride(0)),
        static_cast<std::int64_t>(kv.stride(1)),
        static_cast<std::int64_t>(kv.stride(2))};
    std::vector<std::int64_t> q_output_strides{
        static_cast<std::int64_t>(q_output.stride(0)),
        static_cast<std::int64_t>(q_output.stride(1)),
        static_cast<std::int64_t>(q_output.stride(2))};
    std::vector<std::int64_t> kv_output_strides{
        static_cast<std::int64_t>(kv_output.stride(0)),
        static_cast<std::int64_t>(kv_output.stride(1)),
        static_cast<std::int64_t>(kv_output.stride(2))};
    std::vector<std::int64_t> weight_strides{0, 0, 1};
    std::uint64_t scalar_shape[1] = {};
    std::int64_t scalar_strides[1] = {};
    double eps_value = static_cast<double>(eps);
    std::int64_t width_value = 128;

    auto* q_input_data = static_cast<std::byte*>(q.data());
    auto* q0_input_data = q_input_data;
    auto* q1_input_data = q_input_data + block_width * element_size;
    auto* q_output_data = static_cast<std::byte*>(q_output.data());
    auto* q0_output_data = q_output_data;
    auto* q1_output_data = q_output_data + block_width * element_size;

    ::NineToothedTensor q0_input{q0_input_data, block_sizes.data(),
                                 q_input_strides.data()};
    ::NineToothedTensor q1_input{q1_input_data, block_sizes.data(),
                                 q_input_strides.data()};
    ::NineToothedTensor kv_input{kv.data(), block_sizes.data(),
                                 kv_input_strides.data()};
    ::NineToothedTensor q0_weight{q_weight.data(), block_sizes.data(),
                                  weight_strides.data()};
    ::NineToothedTensor q1_weight{q_weight.data(), block_sizes.data(),
                                  weight_strides.data()};
    ::NineToothedTensor kv_weight_tensor{kv_weight.data(), block_sizes.data(),
                                         weight_strides.data()};
    ::NineToothedTensor eps_tensor{&eps_value, scalar_shape, scalar_strides};
    ::NineToothedTensor q0_output{q0_output_data, block_sizes.data(),
                                  q_output_strides.data()};
    ::NineToothedTensor q1_output{q1_output_data, block_sizes.data(),
                                  q_output_strides.data()};
    ::NineToothedTensor kv_output_tensor{kv_output.data(), block_sizes.data(),
                                         kv_output_strides.data()};
    ::NineToothedTensor width_tensor{&width_value, scalar_shape,
                                     scalar_strides};

    const auto result = launch_infini_ops_ninetoothed_fused_qk_rms_norm(
        static_cast<NineToothedStream>(stream_), q0_input, q1_input, kv_input,
        q0_weight, q1_weight, kv_weight_tensor, eps_tensor, q0_output,
        q1_output, kv_output_tensor, width_tensor, 9);
    assert(result == 0 && "NineToothed `FusedQkRmsNorm` launch failed");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NINETOOTHED_FUSED_QK_RMS_NORM_H_
