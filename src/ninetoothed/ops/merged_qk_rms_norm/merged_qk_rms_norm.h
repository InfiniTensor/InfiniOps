#ifndef INFINI_OPS_NINETOOTHED_MERGED_QK_RMS_NORM_H_
#define INFINI_OPS_NINETOOTHED_MERGED_QK_RMS_NORM_H_

#include <cassert>
#include <cstdint>
#include <vector>

#include "base/merged_qk_rms_norm.h"
#include "data_type.h"
#include "merged_qk_rms_norm/infini_ops_ninetoothed_merged_qk_rms_norm.h"
#include "ninetoothed/tensor.h"

namespace infini::ops {

template <>
class Operator<MergedQkRmsNorm, Device::Type::kNvidia, 9>
    : public MergedQkRmsNorm {
 public:
  using MergedQkRmsNorm::MergedQkRmsNorm;
  using MergedQkRmsNorm::operator();

  static bool supports(const Tensor input, const Tensor weight, float,
                       Tensor output) {
    return input.ndim() == 3 && output.ndim() == 3 &&
           input.shape() == output.shape() && input.size(2) == 128 &&
           input.stride(1) == 128 && input.stride(2) == 1 &&
           output.stride(1) == 128 && output.stride(2) == 1 &&
           input.dtype() == DataType::kBFloat16 &&
           output.dtype() == input.dtype() && weight.dtype() == input.dtype() &&
           weight.ndim() == 2 && weight.size(0) == input.size(1) &&
           weight.size(1) == 128 && weight.stride(0) == 128 &&
           weight.stride(1) == 1;
  }

  void operator()(const Tensor input, const Tensor weight, float epsilon,
                  Tensor output) const override {
    std::vector<std::uint64_t> weight_sizes{
        static_cast<std::uint64_t>(input.size(0)),
        static_cast<std::uint64_t>(input.size(1)),
        static_cast<std::uint64_t>(input.size(2))};
    std::vector<std::int64_t> weight_strides{
        0, static_cast<std::int64_t>(weight.stride(0)),
        static_cast<std::int64_t>(weight.stride(1))};
    std::uint64_t scalar_shape[1] = {};
    std::int64_t scalar_strides[1] = {};
    double epsilon_value = static_cast<double>(epsilon);
    std::int64_t normalized_width = static_cast<std::int64_t>(input.size(2));

    ninetoothed::Tensor input_tensor(input);
    ninetoothed::Tensor weight_tensor(const_cast<void*>(weight.data()),
                                      weight_sizes.data(),
                                      weight_strides.data());
    ninetoothed::Tensor epsilon_tensor(epsilon_value, scalar_shape,
                                       scalar_strides);
    ninetoothed::Tensor output_tensor(output);
    ninetoothed::Tensor width_tensor(normalized_width, scalar_shape,
                                     scalar_strides);

    const auto result = launch_infini_ops_ninetoothed_merged_qk_rms_norm(
        static_cast<NineToothedStream>(stream_), input_tensor, weight_tensor,
        epsilon_tensor, output_tensor, width_tensor);
    assert(result == 0 && "NineToothed `MergedQkRmsNorm` launch failed");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NINETOOTHED_MERGED_QK_RMS_NORM_H_
