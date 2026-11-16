#ifndef INFINI_OPS_TORCH_CONVOLUTION_DIRECT_OUTPUT_H_
#define INFINI_OPS_TORCH_CONVOLUTION_DIRECT_OUTPUT_H_

#include <ATen/Context.h>
#include <ATen/ops/cudnn_convolution.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace infini::ops::torch_conv_detail {

inline bool TryDirectOutput(const at::Tensor& input, const at::Tensor& weight,
                            const std::optional<at::Tensor>& bias,
                            at::IntArrayRef stride, at::IntArrayRef padding,
                            at::IntArrayRef dilation, bool transposed,
                            at::IntArrayRef output_padding, int64_t groups,
                            at::Tensor& out) {
  const auto& context = at::globalContext();
  if (!context.userEnabledCuDNN() || bias.has_value() || transposed ||
      groups <= 0 || input.scalar_type() != at::kFloat ||
      weight.scalar_type() != at::kFloat || out.scalar_type() != at::kFloat ||
      input.dim() != 4 || weight.dim() != 4 || out.dim() != 4 ||
      input.device() != weight.device() || input.device() != out.device() ||
      !input.is_contiguous() || !weight.is_contiguous() ||
      !out.is_contiguous() || input.numel() == 0 || weight.numel() == 0 ||
      out.numel() == 0 || stride.size() != 2 || padding.size() != 2 ||
      dilation.size() != 2 || output_padding.size() != 2 ||
      output_padding[0] != 0 || output_padding[1] != 0) {
    return false;
  }
  for (const auto* tensor :
       {&input, &weight, static_cast<const at::Tensor*>(&out)}) {
    for (const auto extent : tensor->sizes()) {
      if (extent <= 0 || extent > std::numeric_limits<int>::max()) return false;
    }
  }
  if (input.size(0) != out.size(0) || weight.size(0) != out.size(1) ||
      input.size(1) % groups != 0 || weight.size(1) != input.size(1) / groups ||
      weight.size(0) % groups != 0)
    return false;
  for (size_t axis = 0; axis < 2; ++axis) {
    if (stride[axis] <= 0 || stride[axis] > std::numeric_limits<int>::max() ||
        padding[axis] < 0 || padding[axis] > std::numeric_limits<int>::max() ||
        dilation[axis] <= 0 || dilation[axis] > std::numeric_limits<int>::max())
      return false;
    const auto extent = input.size(axis + 2) + 2 * padding[axis] -
                        dilation[axis] * (weight.size(axis + 2) - 1) - 1;
    if (extent < 0 || out.size(axis + 2) != extent / stride[axis] + 1)
      return false;
  }
  const auto overlaps = [](const at::Tensor& left, const at::Tensor& right) {
    const auto a = reinterpret_cast<uintptr_t>(left.data_ptr());
    const auto b = reinterpret_cast<uintptr_t>(right.data_ptr());
    return a <= b ? b - a < left.nbytes() : a - b < right.nbytes();
  };
  if (overlaps(out, input) || overlaps(out, weight)) return false;
  void* output_pointer = out.data_ptr();
  at::cudnn_convolution_out(
      out, input, weight, padding, stride, dilation, groups,
      context.benchmarkCuDNN(),
      context.deterministicCuDNN() || context.deterministicAlgorithms(),
      context.allowTF32CuDNN());
  assert(out.data_ptr() == output_pointer &&
         "`Convolution` must preserve output storage");
  return true;
}

}  // namespace infini::ops::torch_conv_detail

#endif
