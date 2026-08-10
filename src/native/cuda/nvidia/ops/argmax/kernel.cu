#include "native/cuda/nvidia/ops/argmax/kernel.h"

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>

#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/ops/argmax/kernel.cuh"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops {

Operator<Argmax, Device::Type::kNvidia>::Operator(
    const Tensor input, const std::optional<int64_t> dim, const bool keepdim,
    Tensor out)
    : Argmax{input, dim, keepdim, out},
      numel_{input.numel()},
      workspace_size_{DispatchWorkspaceSize(input.dtype(), input.numel())} {
  assert(!dim.has_value() && !keepdim &&
         "NVIDIA `Argmax` currently supports only flattened reduction");
  assert(input.IsContiguous() &&
         "NVIDIA `Argmax` requires contiguous input");
  assert(input.numel() > 0 && input.numel() <= INT_MAX &&
         "NVIDIA `Argmax` input size must fit in a positive `int`");
  assert(out.numel() == 1 && out.dtype() == DataType::kInt64 &&
         "NVIDIA `Argmax` requires one `int64` output value");
  assert(input.device() == out.device() &&
         "NVIDIA `Argmax` input and output must be on the same device");

  auto error = Runtime<Device::Type::kNvidia>::Malloc(&default_workspace_,
                                                       workspace_size_);
  assert(error == cudaSuccess &&
         "NVIDIA `Argmax` failed to allocate workspace");
}

Operator<Argmax, Device::Type::kNvidia>::~Operator() {
  auto error = Runtime<Device::Type::kNvidia>::Free(default_workspace_);
  assert(error == cudaSuccess && "NVIDIA `Argmax` failed to free workspace");
}

std::size_t
Operator<Argmax, Device::Type::kNvidia>::workspace_size_in_bytes() const {
  return workspace_size_;
}

void Operator<Argmax, Device::Type::kNvidia>::operator()(
    const Tensor input, const std::optional<int64_t> dim, const bool keepdim,
    Tensor out) const {
  assert(!dim.has_value() && !keepdim);
  assert(input.shape() == input_shape_ && input.strides() == input_strides_
         && input.dtype() == input_type_);
  assert(out.shape() == out_shape_ && out.strides() == out_strides_
         && out.dtype() == out_type_);

  void* workspace = workspace_ ? workspace_ : default_workspace_;
  auto workspace_size = workspace_ ? workspace_size_in_bytes_ : workspace_size_;
  assert(workspace != nullptr && workspace_size >= workspace_size_ &&
         "NVIDIA `Argmax` received insufficient workspace");
  auto stream = reinterpret_cast<cudaStream_t>(stream_);

  DispatchFunc<Device::Type::kNvidia,
               ConcatType<FloatTypes, ReducedFloatTypes>>(
      input.dtype(),
      [&](auto dtype_tag) {
        using T = typename decltype(dtype_tag)::type;
        argmax_detail::Launch(workspace, workspace_size,
                              static_cast<const T*>(input.data()), numel_,
                              static_cast<int64_t*>(out.data()), stream);
      },
      "NVIDIA Argmax");
}

std::size_t Operator<Argmax, Device::Type::kNvidia>::DispatchWorkspaceSize(
    DataType dtype, std::size_t numel) {
  std::size_t workspace_size = 0;
  DispatchFunc<Device::Type::kNvidia,
               ConcatType<FloatTypes, ReducedFloatTypes>>(
      dtype,
      [&](auto dtype_tag) {
        using T = typename decltype(dtype_tag)::type;
        workspace_size = argmax_detail::WorkspaceSize<T>(numel);
      },
      "NVIDIA Argmax workspace");

  return workspace_size;
}

}  // namespace infini::ops
