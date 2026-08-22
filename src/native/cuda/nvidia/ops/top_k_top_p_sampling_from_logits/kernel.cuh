#ifndef INFINI_OPS_NVIDIA_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_CUH_
#define INFINI_OPS_NVIDIA_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_CUH_

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include "data_type.h"
#include "native/cuda/nvidia/caster.cuh"

namespace infini::ops::top_k_top_p_sampling_from_logits_detail {

constexpr std::size_t Align256(std::size_t size) {
  return (size + 255) & ~std::size_t{255};
}

template <typename T>
struct Workspace {
  int32_t* indices;
  T* sorted_logits;
  int32_t* sorted_indices;
  double* cumulative_probabilities;
  void* temporary_storage;
  std::size_t temporary_storage_size;
};

template <typename T>
std::size_t WorkspaceSize(int vocab_size) {
  std::size_t sort_size = 0;
  auto status = cub::DeviceRadixSort::SortPairsDescending(
      nullptr, sort_size, static_cast<const T*>(nullptr),
      static_cast<T*>(nullptr), static_cast<const int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr), vocab_size, 0, sizeof(T) * 8, nullptr);
  assert(status == cudaSuccess &&
         "`TopKTopPSamplingFromLogits` failed to query CUB sort workspace");

  std::size_t scan_size = 0;
  status = cub::DeviceScan::InclusiveSum(
      nullptr, scan_size, static_cast<const double*>(nullptr),
      static_cast<double*>(nullptr), vocab_size, nullptr);
  assert(status == cudaSuccess &&
         "`TopKTopPSamplingFromLogits` failed to query CUB scan workspace");

  return Align256(sizeof(int32_t) * static_cast<std::size_t>(vocab_size)) +
         Align256(sizeof(T) * static_cast<std::size_t>(vocab_size)) +
         Align256(sizeof(int32_t) * static_cast<std::size_t>(vocab_size)) +
         Align256(sizeof(double) * static_cast<std::size_t>(vocab_size)) +
         Align256(sort_size > scan_size ? sort_size : scan_size);
}

template <typename T>
Workspace<T> PartitionWorkspace(void* workspace, std::size_t workspace_size,
                                int vocab_size) {
  auto* cursor = static_cast<char*>(workspace);
  const auto indices_size =
      Align256(sizeof(int32_t) * static_cast<std::size_t>(vocab_size));
  const auto sorted_logits_size =
      Align256(sizeof(T) * static_cast<std::size_t>(vocab_size));
  const auto sorted_indices_size =
      Align256(sizeof(int32_t) * static_cast<std::size_t>(vocab_size));
  const auto probabilities_size =
      Align256(sizeof(double) * static_cast<std::size_t>(vocab_size));

  auto* indices = reinterpret_cast<int32_t*>(cursor);
  cursor += indices_size;
  auto* sorted_logits = reinterpret_cast<T*>(cursor);
  cursor += sorted_logits_size;
  auto* sorted_indices = reinterpret_cast<int32_t*>(cursor);
  cursor += sorted_indices_size;
  auto* cumulative_probabilities = reinterpret_cast<double*>(cursor);
  cursor += probabilities_size;

  const auto fixed_size = indices_size + sorted_logits_size +
                          sorted_indices_size + probabilities_size;
  assert(workspace_size >= fixed_size &&
         "`TopKTopPSamplingFromLogits` received insufficient workspace");

  return {indices,        sorted_logits,
          sorted_indices, cumulative_probabilities,
          cursor,         workspace_size - fixed_size};
}

__global__ void FillIndicesKernel(int32_t* indices, int vocab_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < vocab_size) indices[index] = index;
}

template <Device::Type kDevice, typename T>
__global__ void LogitsToProbabilitiesKernel(const T* sorted_logits,
                                            double* probabilities,
                                            int vocab_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= vocab_size) return;

  const double maximum =
      Caster<kDevice>::template Cast<double>(sorted_logits[0]);
  const double value =
      Caster<kDevice>::template Cast<double>(sorted_logits[index]);
  probabilities[index] = exp(value - maximum);
}

template <typename Tidx>
__global__ void SampleKernel(Tidx* out, const double* cumulative_probabilities,
                             const int32_t* sorted_indices, int vocab_size,
                             int top_k, double top_p, bool joint_filter,
                             double random_value) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  int keep_count = top_k;
  double retained_probability_mass = cumulative_probabilities[keep_count - 1];
  if (joint_filter) {
    const double top_p_probability_mass =
        top_p * cumulative_probabilities[vocab_size - 1];
    if (top_p_probability_mass < retained_probability_mass) {
      retained_probability_mass = top_p_probability_mass;
    }
  } else if (top_p > 0.0 && top_p < 1.0) {
    const double threshold = top_p * cumulative_probabilities[keep_count - 1];
    for (int i = 0; i < keep_count; ++i) {
      if (cumulative_probabilities[i] >= threshold) {
        keep_count = i + 1;
        break;
      }
    }
    retained_probability_mass = cumulative_probabilities[keep_count - 1];
  }

  const double threshold = random_value * retained_probability_mass;
  int selected = 0;
  while (selected + 1 < keep_count &&
         cumulative_probabilities[selected] < threshold) {
    ++selected;
  }
  *out = static_cast<Tidx>(sorted_indices[selected]);
}

template <Device::Type kDevice, typename T, typename Tidx>
void SampleRow(void* workspace, std::size_t workspace_size, Tidx* out,
               const T* logits, int vocab_size, int top_k, double top_p,
               bool joint_filter, double random_value, cudaStream_t stream) {
  auto partition = PartitionWorkspace<T>(workspace, workspace_size, vocab_size);
  constexpr int kBlockSize = 256;
  const auto grid_size = static_cast<unsigned int>(
      vocab_size / kBlockSize + (vocab_size % kBlockSize != 0));

  FillIndicesKernel<<<grid_size, kBlockSize, 0, stream>>>(partition.indices,
                                                          vocab_size);

  auto temporary_storage_size = partition.temporary_storage_size;
  auto status = cub::DeviceRadixSort::SortPairsDescending(
      partition.temporary_storage, temporary_storage_size, logits,
      partition.sorted_logits, partition.indices, partition.sorted_indices,
      vocab_size, 0, sizeof(T) * 8, stream);
  assert(status == cudaSuccess &&
         "`TopKTopPSamplingFromLogits` CUB radix sort failed");

  LogitsToProbabilitiesKernel<kDevice><<<grid_size, kBlockSize, 0, stream>>>(
      partition.sorted_logits, partition.cumulative_probabilities, vocab_size);

  temporary_storage_size = partition.temporary_storage_size;
  status = cub::DeviceScan::InclusiveSum(
      partition.temporary_storage, temporary_storage_size,
      partition.cumulative_probabilities, partition.cumulative_probabilities,
      vocab_size, stream);
  assert(status == cudaSuccess &&
         "`TopKTopPSamplingFromLogits` CUB inclusive scan failed");

  SampleKernel<Tidx><<<1, 1, 0, stream>>>(
      out, partition.cumulative_probabilities, partition.sorted_indices,
      vocab_size, top_k, top_p, joint_filter, random_value);
}

}  // namespace infini::ops::top_k_top_p_sampling_from_logits_detail

#endif  // INFINI_OPS_NVIDIA_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_CUH_
