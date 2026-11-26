#ifndef INFINI_OPS_HYGON_CAST_KERNEL_H_
#define INFINI_OPS_HYGON_CAST_KERNEL_H_

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "base/cast.h"
#include "native/cuda/hygon/caster.cuh"

namespace infini::ops {
namespace hygon_cast_detail {
template <class Src, class Dst>
__global__ void Convert(const Src* input, Dst* output, std::size_t first,
                        std::size_t count) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    if constexpr (std::is_integral_v<Src> && std::is_integral_v<Dst>) {
      output[first + i] = static_cast<Dst>(input[first + i]);
    } else {
      output[first + i] =
          Caster<Device::Type::kHygon>::template Cast<Dst>(input[first + i]);
    }
  }
}
}  // namespace hygon_cast_detail

template <>
class Operator<Cast, Device::Type::kHygon> : public Cast {
 public:
  static bool supports(const Tensor input, const Tensor out) {
    const bool supported_conversion =
        (FloatType(input.dtype()) && FloatType(out.dtype())) ||
        (input.dtype() == DataType::kInt32 && out.dtype() == DataType::kInt64);
    if (!supported_conversion || input.device().type() != Device::Type::kHygon ||
        input.device() != out.device() || input.shape() != out.shape() ||
        !input.IsContiguous() || !out.IsContiguous())
      return false;

    const auto ip = reinterpret_cast<std::uintptr_t>(input.data());
    const auto op = reinterpret_cast<std::uintptr_t>(out.data());
    const auto size = input.numel();
    if (size == 0) return true;
    const auto iw = Width(input.dtype());
    const auto ow = Width(out.dtype());
    const auto limit = std::numeric_limits<std::uintptr_t>::max();
    if (ip == 0 || op == 0 || ip % iw || op % ow ||
        size > (limit - ip) / iw || size > (limit - op) / ow)
      return false;
    return (ip == op && input.dtype() == out.dtype()) ||
           ip + size * iw <= op || op + size * ow <= ip;
  }

  Operator(const Tensor input, Tensor out) : Cast{Checked(input, out), out} {}
  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, Tensor out) const override {
    if (input.dtype() != input_dtype_ || out.dtype() != out_dtype_ ||
        input.shape() != input_shape_ || out.shape() != out_shape_ ||
        !supports(input, out))
      throw std::invalid_argument("Hygon Cast: unsupported call metadata");
    if (output_size_ == 0 || input.data() == out.data()) return;

    if (input.dtype() == DataType::kInt32) {
      Launch<int32_t, int64_t>(input, out);
      return;
    }
    DispatchFunc<List<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>,
                 List<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>>(
        {static_cast<int64_t>(input_dtype_), static_cast<int64_t>(out_dtype_)},
        [&](auto tag) {
          using Src = TypeMapType<Device::Type::kHygon, ListGet<0>(tag)>;
          using Dst = TypeMapType<Device::Type::kHygon, ListGet<1>(tag)>;
          Launch<Src, Dst>(input, out);
        }, "Hygon Cast");
  }

 private:
  static bool FloatType(DataType dtype) {
    return dtype == DataType::kFloat32 || dtype == DataType::kFloat16 ||
           dtype == DataType::kBFloat16;
  }

  static std::size_t Width(DataType dtype) {
    if (dtype == DataType::kInt64) return 8;
    return dtype == DataType::kFloat32 || dtype == DataType::kInt32 ? 4 : 2;
  }

  static Tensor Checked(const Tensor input, Tensor out) {
    if (!supports(input, out))
      throw std::invalid_argument("Hygon Cast: unsupported dtype, layout or alias");
    return input;
  }

  template <class Src, class Dst>
  void Launch(const Tensor input, Tensor out) const {
    constexpr std::size_t threads = 256;
    constexpr std::size_t capacity = 65535 * threads;
    const auto stream = static_cast<cudaStream_t>(stream_);
    for (std::size_t first = 0; first < output_size_;) {
      const auto count = std::min(capacity, output_size_ - first);
      const auto blocks = static_cast<unsigned>((count + threads - 1) / threads);
      hygon_cast_detail::Convert<Src, Dst><<<blocks, threads, 0, stream>>>(
          static_cast<const Src*>(input.data()), static_cast<Dst*>(out.data()),
          first, count);
      const auto status = cudaGetLastError();
      if (status != cudaSuccess)
        throw std::runtime_error(std::string("Hygon Cast: ") + cudaGetErrorString(status));
      first += count;
    }
  }
};
}  // namespace infini::ops

#endif
