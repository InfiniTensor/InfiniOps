#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "native/cuda/hygon/ops/moe_fused_dense/kernel.h"

#if defined(INFINI_OPS_HYGON_MMAC)
#include "native/cuda/hygon/ops/moe_fused_dense/kernel.cuh"
#endif

namespace infini::ops {
namespace {

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_index) {
    auto status = cudaGetDevice(&previous_device_);
    assert(status == cudaSuccess &&
           "`MoeFusedDense` failed to query the current CUDA device");

    if (previous_device_ != device_index) {
      status = cudaSetDevice(device_index);
      assert(status == cudaSuccess &&
             "`MoeFusedDense` failed to select the input CUDA device");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      const auto status = cudaSetDevice(previous_device_);
      assert(status == cudaSuccess &&
             "`MoeFusedDense` failed to restore the CUDA device");
    }
  }

 private:
  int previous_device_{0};

  bool restore_{false};
};

#if defined(INFINI_OPS_HYGON_MMAC)

// Wavefronts the two streaming kernels are launched with: kWavesPerCu per CU of the device,
// cached per device. Every wavefront owns 8 KB of LDS, so the ones beyond what a CU can host
// would only wait for a slot.
int WavefrontBudget(int device_index) {
  constexpr int kCachedDevices = 64;
  static std::atomic<int> cached[kCachedDevices]{};

  int count = 0;
  if (device_index >= 0 && device_index < kCachedDevices) {
    count = cached[device_index].load(std::memory_order_relaxed);
    if (count != 0) return count;
  }
  const auto status = cudaDeviceGetAttribute(
      &count, cudaDevAttrMultiProcessorCount, device_index);
  assert(status == cudaSuccess &&
         "`MoeFusedDense` failed to query the multiprocessor count");
  if (status != cudaSuccess) count = 1;
  count = std::max(count, 1) * moe_fused_dense_detail::kWavesPerCu;
  if (device_index >= 0 && device_index < kCachedDevices) {
    cached[device_index].store(count, std::memory_order_relaxed);
  }
  return count;
}

// The three kernels for one element type. A block slot is `block_size` rows of `sorted`; a
// work item covers 16 features (w13) or 32 output rows (w2) of one slot.
template <typename T>
void Launch(const void* hidden, const void* w13, const void* w2, uint16_t* act,
            float* pair_out, uint16_t* output, const float* weights,
            const int32_t* ids, const int32_t* sorted, const int32_t* experts,
            const int32_t* total, int tokens, int h, int i, int topk,
            int num_experts, int slots, int waves, cudaStream_t stream) {
  using namespace moe_fused_dense_detail;
  const int groups13 = (i + kBlock - 1) / kBlock;
  const int groups2 = (h + 2 * kBlock - 1) / (2 * kBlock);
  const int items13 = slots * groups13, items2 = slots * groups2;

  W13Kernel<T><<<dim3(std::min(items13, waves)), dim3(kLanes), 0, stream>>>(
      static_cast<const uint16_t*>(hidden),
      static_cast<const uint16_t*>(w13), act, sorted, experts, total, tokens,
      topk, h, i, items13, groups13);
  W2Kernel<T><<<dim3(std::min(items2, waves)), dim3(kLanes), 0, stream>>>(
      act, static_cast<const uint16_t*>(w2), pair_out, weights, sorted, experts,
      total, tokens, topk, h, i, items2, groups2);
  CombineKernel<T><<<dim3(static_cast<unsigned>(tokens)), dim3(256), 0, stream>>>(
      pair_out, ids, output, tokens, topk, h, num_experts);

  const auto status = cudaGetLastError();
  assert(status == cudaSuccess && "`MoeFusedDense` CUDA kernel launch failed");
}

#endif  // INFINI_OPS_HYGON_MMAC

}  // namespace

namespace moe_fused_dense_detail {

bool HasNativeKernels() {
#if defined(INFINI_OPS_HYGON_MMAC)
  // INFINI_OPS_HYGON_MOE_TRITON=1 keeps the Triton JIT kernels in use instead.
  static const bool triton_only = [] {
    const char* value = std::getenv("INFINI_OPS_HYGON_MOE_TRITON");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
  }();
  return !triton_only;
#else
  return false;
#endif
}

}  // namespace moe_fused_dense_detail

#if defined(INFINI_OPS_HYGON_MMAC)

void Operator<MoeFusedDense, Device::Type::kHygon, 0>::Run(
    Tensor output, const Tensor hidden, const Tensor w13, const Tensor w2,
    const Tensor weights, const Tensor ids, const Tensor sorted,
    const Tensor experts, const Tensor total, size_t block_size) const {
  DeviceGuard device_guard{device_index_};

  const auto stream = static_cast<cudaStream_t>(stream_);
  const auto tokens = static_cast<int>(hidden.size(0)),
             h = static_cast<int>(hidden.size(1)), i = static_cast<int>(w2.size(2)),
             topk = static_cast<int>(ids.size(1)),
             slots = static_cast<int>(sorted.numel() / block_size);
  // The workspace holds act[pairs, i] in the element type, then the fp32 expert outputs
  // [pairs, h].
  moe_detail::Workspace(workspace_, workspace_size_in_bytes_,
                        workspace_size_in_bytes(),
                        {output, hidden, w13, w2, weights, ids, sorted, experts,
                         total});
  auto* act = static_cast<uint16_t*>(workspace_);
  auto* pair_out =
      reinterpret_cast<float*>(act + static_cast<size_t>(tokens) * topk * i);

  const int waves = WavefrontBudget(device_index_);
  if (hidden.dtype() == DataType::kBFloat16) {
    Launch<__nv_bfloat16>(hidden.data(), w13.data(), w2.data(), act, pair_out,
                          static_cast<uint16_t*>(output.data()),
                          static_cast<const float*>(weights.data()),
                          static_cast<const int32_t*>(ids.data()),
                          static_cast<const int32_t*>(sorted.data()),
                          static_cast<const int32_t*>(experts.data()),
                          static_cast<const int32_t*>(total.data()), tokens, h, i,
                          topk, static_cast<int>(w13.size(0)), slots, waves,
                          stream);
  } else {
    Launch<half>(hidden.data(), w13.data(), w2.data(), act, pair_out,
                 static_cast<uint16_t*>(output.data()),
                 static_cast<const float*>(weights.data()),
                 static_cast<const int32_t*>(ids.data()),
                 static_cast<const int32_t*>(sorted.data()),
                 static_cast<const int32_t*>(experts.data()),
                 static_cast<const int32_t*>(total.data()), tokens, h, i, topk,
                 static_cast<int>(w13.size(0)), slots, waves, stream);
  }
}

#else

// Never reached: NativeSupports is false without the matrix instructions.
void Operator<MoeFusedDense, Device::Type::kHygon, 0>::Run(
    Tensor, const Tensor, const Tensor, const Tensor, const Tensor,
    const Tensor, const Tensor, const Tensor, const Tensor, size_t) const {
  throw std::logic_error(
      "`MoeFusedDense` native kernels need a Hygon architecture with matrix "
      "instructions");
}

#endif  // INFINI_OPS_HYGON_MMAC

}  // namespace infini::ops
