#ifndef INFINI_OPS_MARS_ARGMAX_KERNEL_H_
#define INFINI_OPS_MARS_ARGMAX_KERNEL_H_

#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>
#include <stdexcept>

#include "base/argmax.h"
#include "native/cuda/mars/runtime_.h"

namespace infini::ops::mars_argmax_detail {
void Launch(const void* input, DataType dtype, std::size_t count, void* out,
            DataType out_dtype, cudaStream_t stream);
}  // namespace infini::ops::mars_argmax_detail

namespace infini::ops {

template <>
class Operator<Argmax, Device::Type::kMars, 0> : public Argmax {
 public:
  static bool supports(const Tensor input, const std::optional<int64_t> dim,
                       const bool keepdim, Tensor out) {
    return input.device().type() == Device::Type::kMars &&
           input.device() == out.device() && input.ndim() == 1 &&
           input.IsContiguous() && out.numel() == 1 && !keepdim &&
           (!dim.has_value() || *dim == 0) &&
           (input.dtype() == DataType::kFloat16 ||
            input.dtype() == DataType::kBFloat16 ||
            input.dtype() == DataType::kFloat32) &&
           (out.dtype() == DataType::kInt32 || out.dtype() == DataType::kInt64);
  }

  Operator(const Tensor input, const std::optional<int64_t> dim,
           const bool keepdim, Tensor out)
      : Argmax(Checked(input, dim, keepdim, out), dim, keepdim, out),
        input_meta_(input),
        out_meta_(out) {}

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, const std::optional<int64_t> dim,
                  const bool keepdim, Tensor out) const override {
    if (!std::equal_to<Tensor>{}(input, input_meta_) ||
        !std::equal_to<Tensor>{}(out, out_meta_) || dim != dim_ ||
        keepdim != keepdim_)
      throw std::invalid_argument(
          "Mars Argmax metadata or alias changed after Make");
    mars_argmax_detail::Launch(
        input.data(), input.dtype(), input.numel(), out.data(), out.dtype(),
        static_cast<cudaStream_t>(stream_ ? stream_ : 0));
  }

 private:
  static Tensor Checked(const Tensor input, const std::optional<int64_t>& dim,
                        bool keepdim, Tensor out) {
    if (!supports(input, dim, keepdim, out))
      throw std::invalid_argument(
          "unsupported Mars Argmax metadata, device or dtype");
    return input;
  }

  Tensor input_meta_;
  Tensor out_meta_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_MARS_ARGMAX_KERNEL_H_
