#ifndef INFINI_OPS_ILUVATAR_ARGMAX_KERNEL_H_
#define INFINI_OPS_ILUVATAR_ARGMAX_KERNEL_H_

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

#include "base/argmax.h"
#include "native/cuda/iluvatar/ops/argmax/kernel.cuh"
#include "native/cuda/iluvatar/runtime_.h"

namespace infini::ops {

template <>
class Operator<Argmax, Device::Type::kIluvatar> : public Argmax {
 public:
  static bool supports(const Tensor input, const std::optional<int64_t> dim,
                       bool keepdim, const Tensor out) {
    return Supports(input, dim, keepdim, out, false);
  }

  Operator(const Tensor input, const std::optional<int64_t> dim,
           const bool keepdim, Tensor out)
      : Argmax(input, dim, keepdim, out) {
    Validate(input, dim, keepdim, out);
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, const std::optional<int64_t> dim,
                  const bool keepdim, Tensor out) const override {
    if (input.shape() != input_shape_ || input.strides() != input_strides_ ||
        input.dtype() != input_type_ || out.shape() != out_shape_ ||
        out.strides() != out_strides_ || out.dtype() != out_type_ ||
        dim != dim_ || keepdim != keepdim_ ||
        out.device().index() != device_index_) {
      throw std::invalid_argument(
          "ILUVATAR Argmax call metadata changed after Make");
    }
    Validate(input, dim, keepdim, out);
    const auto rows =
        static_cast<unsigned>(input.ndim() == 2 ? input.shape()[0] : 1);
    const auto width = static_cast<int>(input.shape().back());
    const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : 0);
    switch (input.dtype()) {
      case DataType::kFloat16:
        Launch(static_cast<const half*>(input.data()), out, rows, width,
               stream);
        break;
      case DataType::kBFloat16:
        Launch(static_cast<const __nv_bfloat16*>(input.data()), out, rows,
               width, stream);
        break;
      case DataType::kFloat32:
        Launch(static_cast<const float*>(input.data()), out, rows, width,
               stream);
        break;
      default:
        throw std::invalid_argument("ILUVATAR Argmax unsupported dtype");
    }
  }

 private:
  static bool Supports(const Tensor input, const std::optional<int64_t> dim,
                       bool keepdim, const Tensor out, bool invocation) {
    if (input.device().type() != Device::Type::kIluvatar ||
        input.device() != out.device() ||
        (input.ndim() != 1 && input.ndim() != 2) || keepdim ||
        (input.ndim() == 1 ? dim.has_value() : dim != 1) ||
        !input.IsContiguous() || !out.IsContiguous() ||
        (input.dtype() != DataType::kFloat16 &&
         input.dtype() != DataType::kBFloat16 &&
         input.dtype() != DataType::kFloat32) ||
        out.dtype() != DataType::kInt64) {
      return false;
    }
    const auto rows = input.ndim() == 2 ? input.shape()[0] : 1;
    const auto width = input.shape().back();
    if (!rows || !width || rows > static_cast<std::size_t>(INT_MAX) ||
        width > static_cast<std::size_t>(INT_MAX) ||
        (input.ndim() == 1 ? out.ndim() != 0
                           : out.ndim() != 1 || out.shape()[0] != rows)) {
      return false;
    }
    if (rows > std::numeric_limits<std::size_t>::max() / width ||
        rows * width >
            std::numeric_limits<std::size_t>::max() / input.element_size() ||
        rows > std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t)) {
      return false;
    }
    if (!invocation) return true;
    if (!input.data() || !out.data() ||
        reinterpret_cast<std::uintptr_t>(input.data()) % input.element_size() ||
        reinterpret_cast<std::uintptr_t>(out.data()) % alignof(std::int64_t))
      return false;
    const auto input_bytes = input.numel() * input.element_size();
    const auto output_bytes = out.numel() * out.element_size();
    const auto in_begin = reinterpret_cast<std::uintptr_t>(input.data());
    const auto out_begin = reinterpret_cast<std::uintptr_t>(out.data());
    if (input_bytes > std::numeric_limits<std::uintptr_t>::max() - in_begin ||
        output_bytes > std::numeric_limits<std::uintptr_t>::max() - out_begin ||
        (in_begin < out_begin + output_bytes &&
         out_begin < in_begin + input_bytes)) {
      return false;
    }
    int current_device = -1;
    const auto status = cudaGetDevice(&current_device);
    return status == cudaSuccess && current_device == input.device().index();
  }

  static void Validate(const Tensor input, const std::optional<int64_t> dim,
                       bool keepdim, const Tensor out) {
    if (!Supports(input, dim, keepdim, out, true))
      throw std::invalid_argument(
          "unsupported ILUVATAR Argmax metadata, storage or current device");
  }

  template <typename T>
  static void Launch(const T* input, Tensor out, unsigned rows, int width,
                     cudaStream_t stream) {
    iluvatar_argmax_detail::RowsKernel<T>
        <<<rows, iluvatar_argmax_detail::kThreads, 0, stream>>>(
            input, static_cast<std::int64_t*>(out.data()), width);
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      throw std::runtime_error(cudaGetErrorString(status));
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ILUVATAR_ARGMAX_KERNEL_H_
