#ifndef INFINI_OPS_NINETOOTHED_FUSED_ADD_RMS_NORM_H_
#define INFINI_OPS_NINETOOTHED_FUSED_ADD_RMS_NORM_H_

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "base/fused_add_rms_norm.h"
#include "data_type.h"
#include "fused_add_rms_norm/infini_ops_ninetoothed_fused_add_rms_norm.h"
#include "ninetoothed/tensor.h"

namespace infini::ops {

template <>
class Operator<FusedAddRmsNorm, Device::Type::kNvidia, 9>
    : public FusedAddRmsNorm {
 public:
  using FusedAddRmsNorm::FusedAddRmsNorm;
  using FusedAddRmsNorm::operator();

  static bool supports(Tensor input, Tensor residual,
                       const std::optional<Tensor> weight, float) {
    return input.ndim() >= 2 && input.ndim() <= 4 && input.size(-1) <= 1024 &&
           input.dtype() == residual.dtype() &&
           ninetoothed::DataTypeIndex(input.dtype()) >= 0 &&
           (!weight.has_value() || weight->dtype() == input.dtype());
  }

  void operator()(Tensor input, Tensor residual,
                  const std::optional<Tensor> weight,
                  float epsilon) const override {
    assert((input.ndim() == 2 || input.ndim() == 3 || input.ndim() == 4) &&
           "NineToothed `FusedAddRmsNorm` currently supports rank-2, rank-3, "
           "and rank-4 tensors");

    std::vector<std::uint64_t> weight_sizes;
    std::vector<std::int64_t> weight_strides;
    double eps_value = static_cast<double>(epsilon);
    std::int64_t num_normalized_elements =
        static_cast<std::int64_t>(input.size(-1));
    std::uint64_t empty_shape[1] = {};
    std::int64_t empty_strides[1] = {};

    weight_sizes.assign(input.shape().begin(), input.shape().end());
    weight_strides.assign(input.ndim(), 0);
    if (weight.has_value()) {
      weight_strides.back() =
          weight->strides().empty() ? 1 : weight->strides().back();
    }

    const int dtype_index = ninetoothed::DataTypeIndex(input.dtype());
    assert(dtype_index >= 0 &&
           "NineToothed `FusedAddRmsNorm` supports only float16, bfloat16, and "
           "float32");

    ninetoothed::Tensor input_tensor(input);
    ninetoothed::Tensor residual_tensor(residual);
    ninetoothed::Tensor weight_tensor(
        weight.has_value() ? const_cast<void*>(weight->data())
                           : const_cast<void*>(input.data()),
        weight_sizes.data(), weight_strides.data());
    int32_t has_weight_value = weight.has_value() ? 1 : 0;
    ninetoothed::Tensor has_weight_tensor(has_weight_value, empty_shape,
                                          empty_strides);
    ninetoothed::Tensor eps_tensor(eps_value, empty_shape, empty_strides);
    ninetoothed::Tensor num_normalized_elements_tensor(
        num_normalized_elements, empty_shape, empty_strides);

    auto result = launch_infini_ops_ninetoothed_fused_add_rms_norm(
        static_cast<NineToothedStream>(stream_), input_tensor, residual_tensor,
        weight_tensor, has_weight_tensor, eps_tensor,
        num_normalized_elements_tensor, static_cast<int>(input.ndim()),
        dtype_index);
    assert(result == 0 && "NineToothed `FusedAddRmsNorm` launch failed");
  }
};

}  // namespace infini::ops

#endif
