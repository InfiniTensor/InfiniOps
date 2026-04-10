#ifndef INFINI_OPS_CUDA_TEMPLATES_REDUCE_TRANSFORM_CUH_
#define INFINI_OPS_CUDA_TEMPLATES_REDUCE_TRANSFORM_CUH_

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>

#include "cuda/caster.cuh"
#include "cuda/kernel_commons.cuh"
#include "cuda/runtime_utils.h"
#include "dispatcher.h"
#include "tensor.h"

namespace infini::ops {

// Generic reduce-then-transform GPU kernel.
//
// One CUDA block processes one logical unit (e.g. one [batch, head] slice).
// The reduction runs over `reduce_dim` elements using CUB `BlockReduce`,
// then the transform writes back `reduce_dim` elements using all threads.
//
// Template parameters:
//   `ReduceOp`    — functor: `TCompute operator()(const TData* ptr, size_t count)`
//                   returns per-thread partial result for BlockReduce::Sum.
//   `TransformOp` — functor: `TData operator()(TData x, TCompute reduced, size_t i)`
//                   applied per element after reduction.
template <unsigned int block_size, Device::Type kDev, typename TCompute,
          typename TData, typename ReduceOp, typename TransformOp>
__global__ void ReduceThenTransformKernel(
    TData* __restrict__ out, int64_t stride_out_batch, int64_t stride_out_head,
    const TData* __restrict__ in, int64_t stride_in_batch,
    int64_t stride_in_head, size_t nhead, size_t reduce_dim,
    ReduceOp reduce_op, TransformOp transform_op) {
  size_t batch_idx = blockIdx.x / nhead;
  size_t head_idx = blockIdx.x % nhead;

  auto out_ptr = out + batch_idx * stride_out_batch + head_idx * stride_out_head;
  auto in_ptr = in + batch_idx * stride_in_batch + head_idx * stride_in_head;

  // Reduction phase: each thread accumulates a partial sum, then block-reduce.
  TCompute partial = reduce_op.template Accumulate<block_size, kDev>(
      in_ptr, reduce_dim);

  using BlockReduce = cub::BlockReduce<TCompute, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  TCompute total = BlockReduce(temp_storage).Sum(partial);

  // Thread 0 post-processes the reduction result and shares via shared memory.
  __shared__ TCompute reduced;

  if (threadIdx.x == 0) {
    reduced = reduce_op.Finalize(total, reduce_dim);
  }

  __syncthreads();

  // Transform phase: all threads apply the transform in parallel.
  for (size_t i = threadIdx.x; i < reduce_dim; i += block_size) {
    out_ptr[i] = transform_op.template Apply<kDev>(in_ptr[i], reduced, i);
  }
}

// Launches a reduce-then-transform kernel with dtype dispatch.
//
// `ReduceOp` and `TransformOp` are host-side structs that carry any extra
// state (weights, epsilon, etc.) and define device-side methods.
template <typename Backend, typename TypeList, typename ReduceOp,
          typename TransformOp>
void LaunchReduceThenTransform(
    void* stream, const Tensor in, Tensor out, size_t batch_size,
    size_t nhead, size_t reduce_dim, DataType dtype,
    const Tensor::Strides& in_strides, const Tensor::Strides& out_strides,
    ReduceOp reduce_op, TransformOp transform_op) {
  auto cuda_stream =
      static_cast<typename Backend::Stream>(stream ? stream : 0);

  auto stride_in_batch = in_strides.size() > 1 ? in_strides[0] : 0;
  auto stride_in_head =
      in_strides.size() > 1 ? in_strides[1] : in_strides[0];
  auto stride_out_batch = out_strides.size() > 1 ? out_strides[0] : 0;
  auto stride_out_head =
      out_strides.size() > 1 ? out_strides[1] : out_strides[0];

  uint32_t num_blocks = static_cast<uint32_t>(batch_size * nhead);
  int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

  DispatchFunc<TypeList, AllCudaBlockSizes>(
      {static_cast<int64_t>(dtype), block_size},
      [&](auto list_tag) {
        using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
        constexpr int kBlockSize = ListGet<1>(list_tag);

        ReduceThenTransformKernel<kBlockSize, Backend::kDeviceType, float, T,
                                  ReduceOp, TransformOp>
            <<<num_blocks, kBlockSize, 0, cuda_stream>>>(
                reinterpret_cast<T*>(out.data()), stride_out_batch,
                stride_out_head, reinterpret_cast<const T*>(in.data()),
                stride_in_batch, stride_in_head, nhead, reduce_dim, reduce_op,
                transform_op);
      },
      "LaunchReduceThenTransform");
}

// ---------- Built-in reduce/transform ops for common patterns ---------------

// Reduce op: mean of squares (for RmsNorm).
struct MeanSquareReduce {
  template <unsigned int block_size, Device::Type kDev, typename TData>
  __device__ __forceinline__ float Accumulate(const TData* ptr,
                                              size_t count) const {
    float ss = 0;

    for (size_t i = threadIdx.x; i < count; i += block_size) {
      float v = Caster<kDev>::template Cast<float>(ptr[i]);
      ss += v * v;
    }

    return ss;
  }

  __device__ __forceinline__ float Finalize(float total,
                                            size_t count) const {
    return rsqrtf(total / static_cast<float>(count) + epsilon);
  }

  float epsilon;
};

// Transform op: multiply by weight and reduced RMS value (for RmsNorm).
struct RmsNormTransform {
  template <Device::Type kDev, typename TData>
  __device__ __forceinline__ TData Apply(TData x, float rms,
                                         size_t i) const {
    return Caster<kDev>::template Cast<TData>(
        Caster<kDev>::template Cast<float>(x) *
        Caster<kDev>::template Cast<float>(weight[i]) * rms);
  }

  const void* weight;
};

}  // namespace infini::ops

#endif
