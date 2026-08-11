#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>

#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/nvidia/ops/top_k_top_p_sampling_from_logits/kernel.cuh"
#include "native/cuda/nvidia/ops/top_k_top_p_sampling_from_logits/kernel.h"

namespace infini::ops {
namespace {

constexpr uint64_t kCounterIncrement = 0x9e3779b97f4a7c15ULL;

uint64_t MixCounter(uint64_t value) {
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

double CounterBasedUniform(uint64_t seed, uint64_t counter) {
  const auto bits = MixCounter(seed + (counter + 1) * kCounterIncrement);
  constexpr double kInverseTwoToThe53 =
      1.0 / static_cast<double>(uint64_t{1} << 53);
  return (static_cast<double>(bits >> 11) + 0.5) * kInverseTwoToThe53;
}

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_index) {
    auto status = cudaGetDevice(&previous_device_);
    assert(status == cudaSuccess &&
           "`TopKTopPSamplingFromLogits` failed to query the current CUDA "
           "device");

    if (previous_device_ != device_index) {
      status = cudaSetDevice(device_index);
      assert(status == cudaSuccess &&
             "`TopKTopPSamplingFromLogits` failed to select the input CUDA "
             "device");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      const auto status = cudaSetDevice(previous_device_);
      assert(status == cudaSuccess &&
             "`TopKTopPSamplingFromLogits` failed to restore the CUDA device");
    }
  }

 private:
  int previous_device_{0};

  bool restore_{false};
};

}  // namespace

Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia, 0>::Operator(
    const Tensor logits, const Tensor top_k, const Tensor top_p,
    const std::optional<Tensor> indices, const std::string filter_apply_order,
    const bool deterministic, const bool check_nan,
    const std::optional<int64_t> seed, const std::optional<int64_t> offset,
    Tensor out)
    : TopKTopPSamplingFromLogits(logits, top_k, top_p, indices,
                                 filter_apply_order, deterministic, check_nan,
                                 seed, offset, out),
      device_index_{logits.device().index()} {
  ValidateSupportedOptions(filter_apply_order, deterministic, check_nan);
  ValidateHostTensor(top_k);
  ValidateHostTensor(top_p);
  ValidateIndices(indices);
  assert(logits.IsContiguous() &&
         "The NVIDIA `TopKTopPSamplingFromLogits` provider requires "
         "contiguous logits");
  assert(out.IsContiguous() &&
         "The NVIDIA `TopKTopPSamplingFromLogits` provider requires "
         "contiguous output");
  assert(out.device() == logits.device() &&
         "The NVIDIA `TopKTopPSamplingFromLogits` provider requires logits "
         "and output on the same device");
  assert(vocab_size_ > 0 &&
         "The NVIDIA `TopKTopPSamplingFromLogits` provider requires a "
         "nonempty vocabulary");
  assert(vocab_size_ <= std::numeric_limits<int>::max() &&
         "The NVIDIA `TopKTopPSamplingFromLogits` provider requires the "
         "vocabulary size to fit in `int`");

  DeviceGuard device_guard{device_index_};
  workspace_size_ = DispatchWorkspaceSize(dtype_, vocab_size_);
}

Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia, 0>::~Operator() {
  if (std::all_of(default_workspace_slots_.begin(),
                  default_workspace_slots_.end(),
                  [](const auto& slot) { return slot.workspace == nullptr; })) {
    return;
  }

  DeviceGuard device_guard{device_index_};
  for (auto& slot : default_workspace_slots_) {
    if (slot.workspace == nullptr) continue;

    if (slot.completion_recorded) {
      const auto status = cudaEventSynchronize(slot.completion);
      assert(
          status == cudaSuccess &&
          "`TopKTopPSamplingFromLogits` failed to synchronize CUDA workspace");
    }

    auto status = cudaFree(slot.workspace);
    assert(status == cudaSuccess &&
           "`TopKTopPSamplingFromLogits` failed to free CUDA workspace");

    status = cudaEventDestroy(slot.completion);
    assert(status == cudaSuccess &&
           "`TopKTopPSamplingFromLogits` failed to destroy CUDA workspace "
           "event");
  }
}

std::size_t Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia,
                     0>::workspace_size_in_bytes() const {
  return workspace_size_;
}

Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia,
         0>::DefaultWorkspaceSlot*
Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia,
         0>::AcquireDefaultWorkspaceSlot(cudaStream_t stream) const {
  for (auto& slot : default_workspace_slots_) {
    if (slot.workspace != nullptr && slot.stream == stream) return &slot;
  }

  for (auto& slot : default_workspace_slots_) {
    if (slot.workspace == nullptr) {
      auto status = cudaMalloc(&slot.workspace, workspace_size_);
      assert(status == cudaSuccess &&
             "`TopKTopPSamplingFromLogits` failed to allocate CUDA workspace");

      status =
          cudaEventCreateWithFlags(&slot.completion, cudaEventDisableTiming);
      assert(status == cudaSuccess &&
             "`TopKTopPSamplingFromLogits` failed to create CUDA workspace "
             "event");
      slot.stream = stream;
      return &slot;
    }
  }

  auto& slot = default_workspace_slots_[next_default_workspace_slot_];
  next_default_workspace_slot_ =
      (next_default_workspace_slot_ + 1) % kDefaultWorkspaceSlotCount;
  if (slot.completion_recorded) {
    const auto status = cudaEventSynchronize(slot.completion);
    assert(status == cudaSuccess &&
           "`TopKTopPSamplingFromLogits` failed while waiting to reuse CUDA "
           "workspace");
    slot.completion_recorded = false;
  }
  slot.stream = stream;
  return &slot;
}

void Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia,
              0>::RecordDefaultWorkspaceUse(DefaultWorkspaceSlot* slot,
                                            cudaStream_t stream) {
  const auto status = cudaEventRecord(slot->completion, stream);
  assert(status == cudaSuccess &&
         "`TopKTopPSamplingFromLogits` failed to record CUDA workspace use");
  slot->completion_recorded = true;
}

void Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia, 0>::operator()(
    const Tensor logits, const Tensor top_k, const Tensor top_p,
    const std::optional<Tensor> indices, const std::string filter_apply_order,
    const bool deterministic, const bool check_nan,
    const std::optional<int64_t> seed, const std::optional<int64_t> offset,
    Tensor out) const {
  ValidateSupportedOptions(filter_apply_order, deterministic, check_nan);
  ValidateHostTensor(top_k);
  ValidateHostTensor(top_p);
  ValidateIndices(indices);
  assert(logits.IsContiguous() && out.IsContiguous() &&
         "The NVIDIA `TopKTopPSamplingFromLogits` provider requires "
         "contiguous logits and output");

  if (batch_size_ == 0) return;

  DeviceGuard device_guard{device_index_};
  const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : nullptr);
  DefaultWorkspaceSlot* default_workspace_slot = nullptr;
  void* workspace = workspace_;
  if (workspace == nullptr) {
    default_workspace_slot = AcquireDefaultWorkspaceSlot(stream);
    workspace = default_workspace_slot->workspace;
  }
  const auto workspace_size =
      workspace_ ? workspace_size_in_bytes_ : workspace_size_;
  assert(workspace != nullptr && workspace_size >= workspace_size_ &&
         "`TopKTopPSamplingFromLogits` received insufficient workspace");

  const uint64_t actual_seed =
      seed.has_value() ? static_cast<uint64_t>(*seed)
                       : static_cast<uint64_t>(std::random_device{}());
  const uint64_t actual_offset = static_cast<uint64_t>(offset.value_or(0));
  const int vocab_size = static_cast<int>(vocab_size_);
  using OutputTypes = List<DataType::kInt32, DataType::kInt64>;

  DispatchFunc<AllFloatTypes, OutputTypes>(
      {static_cast<int64_t>(logits.dtype()), static_cast<int64_t>(out.dtype())},
      [&](auto list_tag) {
        using T = TypeMapType<Device::Type::kNvidia, ListGet<0>(list_tag)>;
        using Tidx = TypeMapType<Device::Type::kNvidia, ListGet<1>(list_tag)>;
        const auto* logits_ptr = static_cast<const T*>(logits.data());
        auto* out_ptr = static_cast<Tidx*>(out.data());

        for (Tensor::Size row = 0; row < batch_size_; ++row) {
          const int64_t logits_row = indices.has_value()
                                         ? ReadIndex(*indices, row)
                                         : static_cast<int64_t>(row);
          assert(logits_row >= 0 &&
                 static_cast<uint64_t>(logits_row) < logits.size(0) &&
                 "The NVIDIA `TopKTopPSamplingFromLogits` provider received "
                 "an out-of-range row index");
          const int64_t requested_top_k = ReadTopK(top_k, row);
          const int normalized_top_k = requested_top_k <= 0
                                           ? vocab_size
                                           : static_cast<int>(std::min<int64_t>(
                                                 requested_top_k, vocab_size));
          const double requested_top_p = ReadTopP(top_p, row);
          const double normalized_top_p =
              requested_top_p > 0.0 && requested_top_p < 1.0 ? requested_top_p
                                                             : 1.0;

          top_k_top_p_sampling_from_logits_detail::SampleRow<
              Device::Type::kNvidia>(
              workspace, workspace_size, out_ptr + row,
              logits_ptr + logits_row * logits.stride(0), vocab_size,
              normalized_top_k, normalized_top_p, filter_apply_order == "joint",
              CounterBasedUniform(actual_seed,
                                  actual_offset + static_cast<uint64_t>(row)),
              stream);
        }
      },
      "Operator<TopKTopPSamplingFromLogits, "
      "Device::Type::kNvidia>::operator()");

  const auto status = cudaGetLastError();
  assert(status == cudaSuccess &&
         "`TopKTopPSamplingFromLogits` CUDA kernel launch failed");
  if (default_workspace_slot != nullptr) {
    RecordDefaultWorkspaceUse(default_workspace_slot, stream);
  }
}

void Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia,
              0>::ValidateSupportedOptions(const std::string&
                                               filter_apply_order,
                                           bool deterministic, bool check_nan) {
  assert(
      (filter_apply_order == "top_k_first" || filter_apply_order == "joint") &&
      "The NVIDIA `TopKTopPSamplingFromLogits` provider supports only "
      "`top_k_first` and `joint`");
  assert(deterministic &&
         "The NVIDIA `TopKTopPSamplingFromLogits` provider supports only the "
         "deterministic path");
  assert(!check_nan &&
         "The NVIDIA `TopKTopPSamplingFromLogits` provider does not support "
         "`check_nan`");
}

void Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia,
              0>::ValidateHostTensor(const Tensor tensor) {
  assert(tensor.device().type() == Device::Type::kCpu &&
         "The NVIDIA `TopKTopPSamplingFromLogits` provider requires host-side "
         "`top_k` and `top_p` tensors");
  assert(tensor.IsContiguous() &&
         "The NVIDIA `TopKTopPSamplingFromLogits` provider requires "
         "contiguous `top_k` and `top_p` tensors");
}

void Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia,
              0>::ValidateIndices(const std::optional<Tensor>& indices) {
  if (!indices.has_value()) return;

  assert(indices->device().type() == Device::Type::kCpu &&
         "The NVIDIA `TopKTopPSamplingFromLogits` provider requires "
         "host-side `indices`");
  assert(indices->IsContiguous() &&
         "The NVIDIA `TopKTopPSamplingFromLogits` provider requires "
         "contiguous `indices`");
}

int64_t Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia,
                 0>::ReadTopK(const Tensor top_k, Tensor::Size row) {
  const auto element_offset = row * top_k.stride(0);
  if (top_k.dtype() == DataType::kInt32) {
    return static_cast<const int32_t*>(top_k.data())[element_offset];
  }
  return static_cast<const int64_t*>(top_k.data())[element_offset];
}

double Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia, 0>::ReadTopP(
    const Tensor top_p, Tensor::Size row) {
  const auto element_offset = row * top_p.stride(0);
  switch (top_p.dtype()) {
    case DataType::kFloat16:
      return static_cast<const Float16*>(top_p.data())[element_offset]
          .ToFloat();
    case DataType::kBFloat16:
      return static_cast<const BFloat16*>(top_p.data())[element_offset]
          .ToFloat();
    case DataType::kFloat32:
      return static_cast<const float*>(top_p.data())[element_offset];
    case DataType::kFloat64:
      return static_cast<const double*>(top_p.data())[element_offset];
    default:
      assert(false &&
             "`TopKTopPSamplingFromLogits` received unsupported `top_p` "
             "dtype");
      return 1.0;
  }
}

int64_t Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia,
                 0>::ReadIndex(const Tensor indices, Tensor::Size row) {
  const auto element_offset = row * indices.stride(0);
  if (indices.dtype() == DataType::kInt32) {
    return static_cast<const int32_t*>(indices.data())[element_offset];
  }
  return static_cast<const int64_t*>(indices.data())[element_offset];
}

std::size_t Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia,
                     0>::DispatchWorkspaceSize(DataType dtype,
                                               Tensor::Size vocab_size) {
  std::size_t workspace_size = 0;
  DispatchFunc<Device::Type::kNvidia, AllFloatTypes>(
      dtype,
      [&](auto tag) {
        using T = typename decltype(tag)::type;
        workspace_size =
            top_k_top_p_sampling_from_logits_detail::WorkspaceSize<T>(
                static_cast<int>(vocab_size));
      },
      "Operator<TopKTopPSamplingFromLogits, "
      "Device::Type::kNvidia>::DispatchWorkspaceSize");
  return workspace_size;
}

}  // namespace infini::ops
