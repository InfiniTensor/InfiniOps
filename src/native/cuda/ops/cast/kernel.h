#ifndef INFINI_OPS_CUDA_CAST_KERNEL_H_
#define INFINI_OPS_CUDA_CAST_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "base/cast.h"
#include "data_type.h"
#include "device.h"
#include "dispatcher.h"
#include "native/cuda/caster.cuh"
#include "native/cuda/device_buffer.h"
#include "native/cuda/ops/cast/kernel.cuh"

namespace infini::ops {

// Elementwise conversion between float32, float16 and bfloat16. Input and
// output may alias only as the same view. Both must be contiguous unless
// `kStridedLayouts` is set, which also accepts strided operands with a
// non-overlapping output.
template <typename Backend, bool kStridedLayouts = false>
class CudaCast : public Cast {
 public:
  static bool supports(const Tensor input, Tensor out) {
    if (input.device().type() != Backend::kDeviceType ||
        input.device() != out.device() || input.shape() != out.shape() ||
        !SupportedDtype(input.dtype()) || !SupportedDtype(out.dtype()))
      return false;
    if constexpr (kStridedLayouts) {
      if (!NonOverlapping(out)) return false;
    }
    std::size_t input_bytes = 0, out_bytes = 0;
    if (!LayoutBytes(input, &input_bytes) || !LayoutBytes(out, &out_bytes) ||
        !ValidAddress(input, input_bytes) || !ValidAddress(out, out_bytes))
      return false;
    int current = -1;
    if (Backend::GetDevice(&current) != Backend::kSuccess ||
        current != input.device().index())
      return false;
    const auto ip = reinterpret_cast<std::uintptr_t>(input.data());
    const auto op = reinterpret_cast<std::uintptr_t>(out.data());
    const bool overlap = input_bytes && out_bytes && ip < op + out_bytes &&
                         op < ip + input_bytes;
    return !overlap || SameView(input, out);
  }

  CudaCast(const Tensor input, Tensor out)
      : Cast{Checked(input, out), out}, device_{input.device()} {
    layout_.contiguous = is_input_contiguous_ && is_out_contiguous_;
    if (layout_.contiguous) return;
    // Strided operands index through shape and strides in device memory.
    layout_.rank = input.ndim();
    const std::size_t shape_bytes = input.ndim() * sizeof(std::size_t);
    const std::size_t strides_bytes = input.ndim() * sizeof(std::ptrdiff_t);
    const std::size_t bytes = shape_bytes + 2 * strides_bytes;
    std::vector<std::byte> metadata(bytes);
    std::memcpy(metadata.data(), input.shape().data(), shape_bytes);
    std::memcpy(metadata.data() + shape_bytes, input.strides().data(),
                strides_bytes);
    std::memcpy(metadata.data() + shape_bytes + strides_bytes,
                out.strides().data(), strides_bytes);
    auto* device = metadata_.Allocate(bytes);
    if (Backend::Memcpy(device, metadata.data(), bytes,
                        Backend::kMemcpyHostToDevice) != Backend::kSuccess)
      throw std::runtime_error("Cast metadata copy failed");
    layout_.shape = reinterpret_cast<const std::size_t*>(device);
    layout_.input_strides =
        reinterpret_cast<const std::ptrdiff_t*>(device + shape_bytes);
    layout_.output_strides = reinterpret_cast<const std::ptrdiff_t*>(
        device + shape_bytes + strides_bytes);
  }

  void operator()(const Tensor input, Tensor out) const override {
    if (input.dtype() != input_dtype_ || out.dtype() != out_dtype_ ||
        input.shape() != input_shape_ || out.shape() != out_shape_ ||
        input.strides() != input_strides_ || out.strides() != out_strides_ ||
        input.device() != device_ || !supports(input, out))
      throw std::invalid_argument(
          "Cast: call metadata or alias changed after Make");
    if (output_size_ == 0 || SameView(input, out)) return;
    const auto stream = static_cast<typename Backend::Stream>(stream_);
    DispatchFunc<
        List<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>,
        List<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>>(
        {static_cast<int64_t>(input_dtype_), static_cast<int64_t>(out_dtype_)},
        [&](auto list_tag) {
          using InT = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using OutT = TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
          constexpr std::size_t cap = 65535u * 256u;
          for (std::size_t first = 0; first < output_size_;) {
            const auto count = std::min(cap, output_size_ - first);
            const auto blocks = static_cast<unsigned>((count + 255) / 256);
            cast_detail::CastKernel<Backend::kDeviceType, InT, OutT>
                <<<blocks, 256, 0, stream>>>(
                    static_cast<const InT*>(input.data()),
                    static_cast<OutT*>(out.data()), first, count, layout_);
            first += count;
          }
        },
        "Cast");
  }

 private:
  static bool SupportedDtype(DataType type) {
    return type == DataType::kFloat32 || type == DataType::kFloat16 ||
           type == DataType::kBFloat16;
  }

  static std::size_t Width(DataType type) {
    return type == DataType::kFloat32 ? 4u : 2u;
  }

  static bool LayoutBytes(const Tensor t, std::size_t* bytes) {
    if constexpr (kStridedLayouts) return SpanBytes(t, bytes);
    return ContiguousBytes(t, bytes);
  }

  static bool ContiguousBytes(const Tensor t, std::size_t* bytes) {
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
    std::size_t count = 1;
    for (std::size_t axis = t.ndim(); axis > 0; --axis) {
      const auto size = t.size(axis - 1);
      if (size > 1 && t.stride(axis - 1) != static_cast<std::ptrdiff_t>(count))
        return false;
      if (size > limit / count) return false;
      count *= size;
    }
    *bytes = count * Width(t.dtype());
    return true;
  }

  // Bytes from the first to the last element; overflow-checked.
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
          "Cast: unsupported dtype/device/layout or alias");
    return input;
  }

  Device device_;

  // `shape` stays null for contiguous operands.
  cast_detail::Layout layout_{};

  CudaDeviceBuffer<Backend> metadata_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CAST_KERNEL_H_
