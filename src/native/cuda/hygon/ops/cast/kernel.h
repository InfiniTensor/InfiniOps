#ifndef INFINI_OPS_HYGON_CAST_KERNEL_H_
#define INFINI_OPS_HYGON_CAST_KERNEL_H_

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "base/cast.h"
#include "native/cuda/hygon/caster.cuh"

namespace infini::ops {
namespace hygon_cast_detail {
struct Layout {
  size_t rank{};
  size_t shape[16]{};
  ptrdiff_t input_strides[16]{};
  ptrdiff_t output_strides[16]{};
  bool contiguous{};
};
template <typename InT, typename OutT>
__global__ void Convert(const InT* input, OutT* out, std::size_t first,
                        std::size_t count, Layout layout) {
  const auto local =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local >= count) return;
  size_t input_offset = first + local, output_offset = input_offset;
  if (!layout.contiguous) {
    size_t remaining = input_offset;
    input_offset = output_offset = 0;
    for (size_t axis = layout.rank; axis > 0; --axis) {
      const auto coordinate = remaining % layout.shape[axis - 1];
      remaining /= layout.shape[axis - 1];
      input_offset += coordinate * layout.input_strides[axis - 1];
      output_offset += coordinate * layout.output_strides[axis - 1];
    }
  }
  out[output_offset] =
      Caster<Device::Type::kHygon>::template Cast<OutT>(input[input_offset]);
}
}  // namespace hygon_cast_detail

template <>
class Operator<Cast, Device::Type::kHygon> : public Cast {
 public:
  static bool supports(const Tensor input, Tensor out) {
    if (input.device().type() != Device::Type::kHygon ||
        input.device() != out.device() || input.shape() != out.shape() ||
        !SupportedDtype(input.dtype()) || !SupportedDtype(out.dtype()) ||
        input.ndim() > 16 || !NonOverlapping(out))
      return false;
    std::size_t input_bytes = 0, out_bytes = 0;
    if (!SpanBytes(input, &input_bytes) || !SpanBytes(out, &out_bytes) ||
        !ValidAddress(input, input_bytes) || !ValidAddress(out, out_bytes))
      return false;
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess ||
        current != input.device().index())
      return false;
    const auto ip = reinterpret_cast<std::uintptr_t>(input.data());
    const auto op = reinterpret_cast<std::uintptr_t>(out.data());
    const bool overlap = input_bytes && out_bytes && ip < op + out_bytes &&
                         op < ip + input_bytes;
    return !overlap || SameView(input, out);
  }

  Operator(const Tensor input, Tensor out)
      : Cast{Checked(input, out), out}, device_{input.device()} {
    layout_.rank = input.ndim();
    layout_.contiguous = input.IsContiguous() && out.IsContiguous();
    for (size_t axis = 0; axis < input.ndim(); ++axis) {
      layout_.shape[axis] = input.size(axis);
      layout_.input_strides[axis] = input.stride(axis);
      layout_.output_strides[axis] = out.stride(axis);
    }
  }

  void operator()(const Tensor input, Tensor out) const override {
    if (input.dtype() != input_dtype_ || out.dtype() != out_dtype_ ||
        input.shape() != input_shape_ || out.shape() != out_shape_ ||
        input.strides() != input_strides_ || out.strides() != out_strides_ ||
        input.device() != device_ || !supports(input, out))
      throw std::invalid_argument(
          "HYGON Cast: call metadata or alias changed after Make");
    if (output_size_ == 0 || SameView(input, out)) return;
    const auto stream = static_cast<cudaStream_t>(stream_);
    DispatchFunc<
        List<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>,
        List<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>>(
        {static_cast<int64_t>(input_dtype_), static_cast<int64_t>(out_dtype_)},
        [&](auto list_tag) {
          using InT = TypeMapType<Device::Type::kHygon, ListGet<0>(list_tag)>;
          using OutT = TypeMapType<Device::Type::kHygon, ListGet<1>(list_tag)>;
          constexpr std::size_t cap = 65535u * 256u;
          for (std::size_t first = 0; first < output_size_;) {
            const auto count = std::min(cap, output_size_ - first);
            const auto blocks = static_cast<unsigned>((count + 255) / 256);
            hygon_cast_detail::Convert<InT, OutT><<<blocks, 256, 0, stream>>>(
                static_cast<const InT*>(input.data()),
                static_cast<OutT*>(out.data()), first, count, layout_);
            const auto status = cudaGetLastError();
            if (status != cudaSuccess)
              throw std::runtime_error(
                  std::string("HYGON Cast: launch failed: ") +
                  cudaGetErrorString(status));
            first += count;
          }
        },
        "HYGON Cast");
  }

 private:
  static bool SupportedDtype(DataType type) {
    return type == DataType::kFloat32 || type == DataType::kFloat16 ||
           type == DataType::kBFloat16;
  }

  static std::size_t Width(DataType type) {
    return type == DataType::kFloat32 ? 4u : 2u;
  }

  static bool SpanBytes(const Tensor t, std::size_t* bytes) {
    if (t.shape().size() != t.strides().size()) return false;
    bool empty = false;
    for (std::size_t axis = 0; axis < t.ndim(); ++axis) {
      if (t.stride(axis) < 0) return false;
      empty = empty || t.size(axis) == 0;
    }
    if (empty) {
      *bytes = 0;
      return true;
    }
    const auto limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
        Width(t.dtype());
    std::size_t span = 1;
    for (std::size_t axis = 0; axis < t.ndim(); ++axis) {
      const auto size = t.size(axis);
      const auto stride = static_cast<size_t>(t.stride(axis));
      if (size > 1 && stride > (limit - span) / (size - 1)) return false;
      span += (size - 1) * stride;
    }
    *bytes = span * Width(t.dtype());
    return true;
  }

  static bool NonOverlapping(const Tensor tensor) {
    std::vector<std::pair<size_t, size_t>> axes;
    for (size_t axis = 0; axis < tensor.ndim(); ++axis) {
      if (tensor.stride(axis) < 0) return false;
      if (tensor.size(axis) > 1)
        axes.emplace_back(tensor.stride(axis), tensor.size(axis));
    }
    std::sort(axes.begin(), axes.end());
    size_t extent = 1;
    for (const auto& [stride, size] : axes) {
      if (stride < extent ||
          (size - 1) > (std::numeric_limits<size_t>::max() - extent) / stride)
        return false;
      extent += (size - 1) * stride;
    }
    return true;
  }

  static bool ValidAddress(const Tensor t, std::size_t bytes) {
    if (bytes == 0) return true;
    const auto start = reinterpret_cast<std::uintptr_t>(t.data());
    return start && start % Width(t.dtype()) == 0 &&
           start <= std::numeric_limits<std::uintptr_t>::max() - bytes;
  }

  static bool SameView(const Tensor input, const Tensor out) {
    return input.dtype() == out.dtype() && input.data() == out.data() &&
           input.shape() == out.shape() && input.strides() == out.strides();
  }

  static Tensor Checked(const Tensor input, const Tensor out) {
    if (!supports(input, out))
      throw std::invalid_argument(
          "HYGON Cast: unsupported dtype/device/layout or alias");
    return input;
  }

  Device device_;
  hygon_cast_detail::Layout layout_;
};
}  // namespace infini::ops
#endif
