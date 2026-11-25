#ifndef INFINI_OPS_HYGON_CLOVER_LOCAL_MEMORY_KERNEL_H_
#define INFINI_OPS_HYGON_CLOVER_LOCAL_MEMORY_KERNEL_H_

#include <cuda_runtime.h>

#include <array>
#include <climits>
#include <cstdint>
#include <mutex>

#include <infini/rt/hygon/data_type_.h>

#include "base/clover_local_memory.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/ops/clover_local_memory/mma.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/clover_local_memory/kernel.h"

namespace infini::ops {

namespace hygon_clover_local_memory_detail {

// Whether the tensor-core kernel can take the call. It reads the q, k and
// value rows and the state columns as 16-byte vectors, takes the per-head
// scalars in the element type, and maps shared q/k heads with a shift.
inline bool Eligible(const Tensor q, const Tensor k, const Tensor v,
                     const Tensor g, const Tensor state, const Tensor indices,
                     const Tensor valid, const Tensor logit_scale,
                     const Tensor history_bias, const Tensor history_c,
                     std::size_t block_size, const Tensor out) {
  const auto dtype = q.dtype();
  if (dtype != DataType::kFloat16 && dtype != DataType::kBFloat16)
    return false;
  const auto num_v_heads = static_cast<int>(v.size(2));
  const auto num_q_heads = static_cast<int>(q.size(2));
  const auto num_k_heads = static_cast<int>(k.size(2));
  if (q.size(3) != 128 || k.size(3) != 128 || v.size(3) != 128 ||
      state.size(2) != 128 || state.size(3) != 128 ||
      state.size(1) != v.size(2) || block_size == 0 || block_size > 32 ||
      q.size(1) != k.size(1) || q.size(1) != v.size(1))
    return false;
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
  const auto ratio = [num_v_heads](int heads) {
    return heads > 0 && num_v_heads % heads == 0 &&
           ((num_v_heads / heads) & (num_v_heads / heads - 1)) == 0;
  };
  if (!ratio(num_q_heads) || !ratio(num_k_heads)) return false;
  // Whole 16-byte vectors: 8 elements of the row type per unit, the base
  // aligned, and every outer stride keeping the units aligned.
  const auto rows = [&dtype](const Tensor t, std::size_t unit) {
    if (t.ndim() == 0 || t.strides().back() != 1 ||
        reinterpret_cast<std::uintptr_t>(t.data()) % 16 != 0)
      return false;
    for (std::size_t axis = 0; axis + 1 < t.ndim(); ++axis) {
      const auto stride = t.strides()[axis];
      if (stride < 0 || stride % static_cast<std::ptrdiff_t>(unit) != 0)
        return false;
    }
    return true;
  };
  return rows(q, 8) && rows(k, 8) && rows(v, 8) && rows(state, 4) &&
         rows(out, 1);
}

// One tiling per request count, from a measured matrix (1 .. 128 requests,
// 16/32/64 columns): one or two requests want the narrow tiling's extra
// blocks per head, every larger batch the wide one's lower staging repeat;
// 16 columns never won.
inline int SelectTiling(std::size_t requests) {
  return requests <= 2 ? 0 : 1;
}

inline int RatioShift(int num_v_heads, int num_heads) {
  int ratio = num_v_heads / num_heads, shift = 0;
  while (ratio > 1) {
    ratio >>= 1;
    ++shift;
  }
  return shift;
}

template <typename Data, typename Index, int kCols, int kThreads>
bool Launch(const Tensor q, const Tensor k, const Tensor v, const Tensor g,
            const Tensor state, const Tensor indices, const Tensor valid,
            const Tensor logit_scale, const Tensor history_bias,
            const Tensor history_c, std::size_t block_size, Tensor out,
            cudaStream_t stream) {
  dim3 grid(128 / kCols, static_cast<unsigned>(indices.numel()),
            static_cast<unsigned>(v.size(2)));
  CloverLocalMemoryMmaKernel<Data, Index, kCols, kThreads>
      <<<grid, kThreads, 0, stream>>>(
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
          q.stride(1), q.stride(2), k.stride(1), k.stride(2), v.stride(1),
          v.stride(2), g.stride(1), g.stride(2), state.stride(0),
          state.stride(1), state.stride(2), out.stride(1), out.stride(2),
          out.stride(3), 1.0f / sqrtf(128.0f), 32.0f,
          static_cast<int>(block_size), static_cast<int>(state.size(0)),
          RatioShift(static_cast<int>(v.size(2)), static_cast<int>(q.size(2))),
          RatioShift(static_cast<int>(v.size(2)), static_cast<int>(k.size(2))));
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace hygon_clover_local_memory_detail

// Hygon takes the MMAC tensor-core kernel where the call's layout allows
// it; every other layout keeps the shared SIMT kernel.
template <>
class Operator<CloverLocalMemory, Device::Type::kHygon>
    : public CudaCloverLocalMemory<Runtime<Device::Type::kHygon>> {
 public:
  using Base = CudaCloverLocalMemory<Runtime<Device::Type::kHygon>>;
  using Base::Base;

  void operator()(const Tensor q, const Tensor k, const Tensor v,
                  const Tensor g, const Tensor state, const Tensor indices,
                  const Tensor valid, const Tensor logit_scale,
                  const Tensor history_bias, const Tensor history_c,
                  size_t block_size, Tensor out) const override {
    namespace detail = hygon_clover_local_memory_detail;
    const auto requests = indices.numel();
    if (requests != 0 && q.size(1) != 0 &&
        detail::Eligible(q, k, v, g, state, indices, valid, logit_scale,
                         history_bias, history_c, block_size, out)) {
      const auto stream =
          static_cast<cudaStream_t>(stream_ ? stream_ : 0);
      bool launched = false;
      const int tiling = detail::SelectTiling(requests);
      DispatchFunc<
          List<DataType::kFloat16, DataType::kBFloat16>,
          List<DataType::kInt32, DataType::kInt64>>(
          {static_cast<int64_t>(q.dtype()),
           static_cast<int64_t>(indices.dtype())},
          [&](auto list_tag) {
            using Data = TypeMapType<Device::Type::kHygon, ListGet<0>(list_tag)>;
            using Index = TypeMapType<Device::Type::kHygon, ListGet<1>(list_tag)>;
            if (tiling == 0)
              launched = detail::Launch<Data, Index, 32, 256>(
                  q, k, v, g, state, indices, valid, logit_scale,
                  history_bias, history_c, block_size, out, stream);
            else
              launched = detail::Launch<Data, Index, 64, 512>(
                  q, k, v, g, state, indices, valid, logit_scale,
                  history_bias, history_c, block_size, out, stream);
          },
          "HygonCloverLocalMemoryMma::operator()");
      if (launched) return;
    }
    Base::operator()(q, k, v, g, state, indices, valid, logit_scale,
                     history_bias, history_c, block_size, out);
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_CLOVER_LOCAL_MEMORY_KERNEL_H_
