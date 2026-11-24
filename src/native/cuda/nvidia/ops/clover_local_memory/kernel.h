#ifndef INFINI_OPS_Nvidia_CLOVER_LOCAL_MEMORY_KERNEL_H_
#define INFINI_OPS_Nvidia_CLOVER_LOCAL_MEMORY_KERNEL_H_

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <mutex>

#include "base/clover_local_memory.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/nvidia/ops/clover_local_memory/kernel.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/clover_local_memory/kernel.h"

namespace infini::ops {

namespace nvidia_clover_local_memory_detail {

struct DeviceInfo {
  int multiprocessors = 0;  // zero: no tensor-core kernel on this device
};

// Cached per device: the attribute queries cost more than the launch.
inline DeviceInfo Device(int index) {
  static std::mutex mutex;
  static std::array<DeviceInfo, 64> cache;
  static std::array<bool, 64> queried{};
  const auto query = [index] {
    int count = 0;
    int major = 0;
    DeviceInfo info;
    // The kernel needs mma.sync.m16n8k16 for bf16, which starts at sm_80.
    if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, index) ==
            cudaSuccess &&
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               index) == cudaSuccess &&
        major >= 8)
      info = DeviceInfo{count};
    return info;
  };
  if (index < 0 || index >= static_cast<int>(queried.size())) return query();
  std::lock_guard<std::mutex> lock(mutex);
  if (!queried[index]) {
    cache[index] = query();
    queried[index] = true;
  }
  return cache[index];
}

// Whole 16-byte rows: the innermost axis is contiguous, the base and every
// outer stride keep the row starts aligned, and no index or stride exceeds
// 32 bits.
inline bool VectorRows(const Tensor& tensor, std::size_t unit) {
  if (tensor.ndim() == 0 || tensor.strides().back() != 1 ||
      reinterpret_cast<std::uintptr_t>(tensor.data()) % 16 != 0 ||
      tensor.numel() > static_cast<std::size_t>(INT32_MAX))
    return false;
  for (std::size_t axis = 0; axis + 1 < tensor.ndim(); ++axis) {
    const auto stride = tensor.strides()[axis];
    if (stride < 0 || stride % static_cast<std::ptrdiff_t>(unit) != 0 ||
        stride > INT32_MAX)
      return false;
  }
  return true;
}

inline bool NarrowStrides(const Tensor& tensor) {
  for (const auto stride : tensor.strides())
    if (stride < 0 || stride > INT32_MAX) return false;
  return tensor.numel() <= static_cast<std::size_t>(INT32_MAX);
}

// Whether the tensor-core kernel can take the call. It reads the rows of q,
// k, v and the state as 16-byte vectors, takes the per-head scalars in the
// element type and maps shared q/k heads with a shift.
inline bool Eligible(const Tensor q, const Tensor k, const Tensor v, const Tensor g,
              const Tensor state, const Tensor indices, const Tensor valid,
              const Tensor logit_scale, const Tensor history_bias,
              const Tensor history_c, std::size_t block_size,
              const Tensor out) {
  const auto dtype = q.dtype();
  if (dtype != DataType::kFloat16 && dtype != DataType::kBFloat16)
    return false;
  const auto num_v_heads = static_cast<int>(v.size(2));
  const auto num_q_heads = static_cast<int>(q.size(2));
  const auto num_k_heads = static_cast<int>(k.size(2));
  if (q.size(3) != kDim || k.size(3) != kDim || v.size(3) != kDim ||
      state.size(2) != kDim || state.size(3) != kDim ||
      state.size(1) != v.size(2) || block_size == 0 || block_size > kRows ||
      q.size(1) != k.size(1) || q.size(1) != v.size(1))
    return false;
  // Rows of the requests, [request * block_size, +block_size) each.
  if (indices.numel() > q.size(1) / block_size) return false;
  // The scalars are read in the element type; fp32 scalars stay on the
  // shared kernel, which converts them.
  if (k.dtype() != dtype || v.dtype() != dtype || out.dtype() != dtype ||
      logit_scale.dtype() != dtype || history_bias.dtype() != dtype ||
      history_c.dtype() != dtype || g.dtype() != DataType::kFloat32 ||
      state.dtype() != DataType::kFloat32 ||
      valid.dtype() != DataType::kInt8 ||
      (indices.dtype() != DataType::kInt32 &&
       indices.dtype() != DataType::kInt64))
    return false;
  if (!PowerOfTwoRatio(num_v_heads, num_q_heads) ||
      !PowerOfTwoRatio(num_v_heads, num_k_heads))
    return false;
  // 16-bit rows hold 8 elements in 16 bytes, fp32 rows 4.
  return VectorRows(q, 8) && VectorRows(k, 8) && VectorRows(v, 8) &&
         VectorRows(state, 4) && NarrowStrides(g) && NarrowStrides(out) &&
         NarrowStrides(indices);
}

// Which tiling serves the call: 0 for 32, 1 for 64 and 2 for 128 value
// columns per block. A block is bound by latency, so a small batch wants
// many blocks; once the blocks outnumber the SMs, fewer and wider ones
// repeat less of the work per head.
inline int SelectTiling(std::size_t requests, int multiprocessors) {
  if (requests <= 1) return 0;  // four blocks per head spread one request
  if (requests <= 3) return 1;  // two blocks per head
  if (requests * 16 <= static_cast<std::size_t>(multiprocessors))
    return 2;  // one block per head, one per SM
  return 1;
}

template <typename Data, typename Index, int kCols, int kThreads,
          int kMinBlocks>
bool Launch(const Tensor q, const Tensor k, const Tensor v, const Tensor g,
            const Tensor state, const Tensor indices, const Tensor valid,
            const Tensor logit_scale, const Tensor history_bias,
            const Tensor history_c, std::size_t block_size, Tensor out,
            cudaStream_t stream) {
  using Tiling = Tiling<kCols, kThreads>;
  auto* const kernel =
      CloverLocalMemoryMmaKernel<Data, Index, kCols, kThreads, kMinBlocks>;
  // Beyond the 48 KB default a kernel has to opt into its dynamic shared
  // memory size, once per function.
  static const bool shared_ready =
      Tiling::kSmemBytes <= 48 * 1024 ||
      cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           Tiling::kSmemBytes) == cudaSuccess;
  if (!shared_ready) return false;
  dim3 grid(static_cast<unsigned>(kDim / kCols),
            static_cast<unsigned>(indices.numel()),
            static_cast<unsigned>(v.size(2)));
  kernel<<<grid, kThreads, Tiling::kSmemBytes, stream>>>(
      reinterpret_cast<const Data*>(q.data()),
      reinterpret_cast<const Data*>(k.data()),
      reinterpret_cast<const Data*>(v.data()),
      reinterpret_cast<const float*>(g.data()),
      reinterpret_cast<const float*>(state.data()),
      reinterpret_cast<const Index*>(indices.data()),
      reinterpret_cast<const int8_t*>(valid.data()),
      reinterpret_cast<const Data*>(logit_scale.data()),
      reinterpret_cast<const Data*>(history_bias.data()),
      reinterpret_cast<const Data*>(history_c.data()),
      reinterpret_cast<Data*>(out.data()),
      // Layout is [1, tokens, heads, dim]; the batch dim is 1.
      static_cast<int64_t>(q.stride(1)),
      static_cast<int64_t>(q.stride(2)),
      static_cast<int64_t>(k.stride(1)),
      static_cast<int64_t>(k.stride(2)),
      static_cast<int64_t>(v.stride(1)),
      static_cast<int64_t>(v.stride(2)),
      static_cast<int64_t>(g.stride(1)),
      static_cast<int64_t>(g.stride(2)),
      static_cast<int64_t>(state.stride(0)),
      static_cast<int64_t>(state.stride(1)),
      static_cast<int64_t>(state.stride(2)),
      static_cast<int64_t>(out.stride(1)),
      static_cast<int64_t>(out.stride(2)),
      static_cast<int64_t>(out.stride(3)),
      1.0f / sqrtf(128.0f),  // KEY_DIM == VALUE_DIM == 128
      32.0f,  // the reference sizes the history bucket against a full block
      static_cast<int>(block_size), static_cast<int>(state.size(0)),
      RatioShift(static_cast<int>(v.size(2)), static_cast<int>(q.size(2))),
      RatioShift(static_cast<int>(v.size(2)), static_cast<int>(k.size(2))));
  // A failed launch leaves the shared kernel to report or run the call.
  return cudaGetLastError() == cudaSuccess;
}

template <typename Data, typename Index>
bool Launch(int tiling, const Tensor q, const Tensor k, const Tensor v,
            const Tensor g, const Tensor state, const Tensor indices,
            const Tensor valid, const Tensor logit_scale,
            const Tensor history_bias, const Tensor history_c,
            std::size_t block_size, Tensor out, cudaStream_t stream) {
  switch (tiling) {
    case 0:
      return Launch<Data, Index, 32, 128, 1>(q, k, v, g, state, indices, valid,
                                             logit_scale, history_bias,
                                             history_c, block_size, out,
                                             stream);
    case 1:
      return Launch<Data, Index, 64, 256, 3>(q, k, v, g, state, indices, valid,
                                             logit_scale, history_bias,
                                             history_c, block_size, out,
                                             stream);
    default:
      return Launch<Data, Index, 128, 512, 2>(q, k, v, g, state, indices,
                                              valid, logit_scale, history_bias,
                                              history_c, block_size, out,
                                              stream);
  }
}

}  // namespace nvidia_clover_local_memory_detail

// NVIDIA takes the tensor-core kernel on sm_80 and later, where it mirrors
// the Triton reference operation for operation. Every layout it cannot read
// keeps the shared SIMT kernel.
template <>
class Operator<CloverLocalMemory, Device::Type::kNvidia>
    : public CudaCloverLocalMemory<Runtime<Device::Type::kNvidia>> {
 public:
  using Base = CudaCloverLocalMemory<Runtime<Device::Type::kNvidia>>;
  using Base::Base;

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor g, const Tensor state, const Tensor indices,
                  const Tensor valid, const Tensor logit_scale,
                  const Tensor history_bias, const Tensor history_c,
                  size_t block_size, Tensor out) const override {
    namespace detail = nvidia_clover_local_memory_detail;
    const auto requests = indices.numel();
    const auto device = detail::Device(q.device().index());
    if (requests != 0 && q.size(1) != 0 && device.multiprocessors != 0 &&
        detail::Eligible(q, k, v, g, state, indices, valid, logit_scale,
                         history_bias, history_c, block_size, out)) {
      const auto stream =
          static_cast<cudaStream_t>(stream_ ? stream_ : 0);
      bool launched = false;
      DispatchFunc<ReducedFloatTypes, List<DataType::kInt32, DataType::kInt64>>(
          {static_cast<int64_t>(q.dtype()),
           static_cast<int64_t>(indices.dtype())},
          [&](auto list_tag) {
            using Data =
                TypeMapType<Runtime<Device::Type::kNvidia>::kDeviceType,
                            ListGet<0>(list_tag)>;
            using Index =
                TypeMapType<Runtime<Device::Type::kNvidia>::kDeviceType,
                            ListGet<1>(list_tag)>;
            launched = detail::Launch<Data, Index>(
                detail::SelectTiling(requests, device.multiprocessors), q, k,
                v, g, state, indices, valid, logit_scale, history_bias,
                history_c, block_size, out, stream);
          },
          "NvidiaCloverLocalMemory::operator()");
      if (launched) return;
    }
    Base::operator()(q, k, v, g, state, indices, valid, logit_scale,
                     history_bias, history_c, block_size, out);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_Nvidia_CLOVER_LOCAL_MEMORY_KERNEL_H_
