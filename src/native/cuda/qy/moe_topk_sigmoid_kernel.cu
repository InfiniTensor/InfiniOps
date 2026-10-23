#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "native/cuda/qy/data_type_.h"
#include "native/cuda/qy/moe_topk_sigmoid_kernel.h"

namespace infini::ops::qy {
namespace {

constexpr int kMaxTopk = 64;

template <typename T>
__device__ __forceinline__ float toFloat(T v);

template <>
__device__ __forceinline__ float toFloat<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ float toFloat<half>(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float toFloat<cuda_bfloat16>(cuda_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float v);

template <>
__device__ __forceinline__ float fromFloat<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ half fromFloat<half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ cuda_bfloat16 fromFloat<cuda_bfloat16>(float v) {
  return __float2bfloat16(v);
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Load correction bias element; convert to float on device (no host cast).
__device__ __forceinline__ float loadBias(const void *__restrict__ bias,
                                          int bias_dtype_tag, int expert) {
  using infini::rt::DataType;
  switch (static_cast<DataType>(bias_dtype_tag)) {
    case DataType::kFloat32:
      return reinterpret_cast<const float *>(bias)[expert];
    case DataType::kFloat16:
      return __half2float(reinterpret_cast<const half *>(bias)[expert]);
    case DataType::kBFloat16:
      return __bfloat162float(
          reinterpret_cast<const cuda_bfloat16 *>(bias)[expert]);
    default:
      return 0.0f;
  }
}

// CUB-free per-row top-k: grid-stride over tokens, block=128.
// Compute in float; write WeightT / IndexT at the end (after optional renorm).
template <typename GatingT, typename WeightT, typename IndexT>
__global__ void moeTopkSigmoidKernel(const GatingT *__restrict__ gating,
                                     WeightT *__restrict__ topk_weights,
                                     IndexT *__restrict__ topk_indices,
                                     const void *__restrict__ correction_bias,
                                     int bias_dtype_tag, int num_tokens,
                                     int num_experts, int topk,
                                     int renormalize) {
  __shared__ float s_val[128];
  __shared__ int s_idx[128];
  __shared__ int selected[kMaxTopk];
  __shared__ float s_weights[kMaxTopk];

  for (int token = static_cast<int>(blockIdx.x); token < num_tokens;
       token += static_cast<int>(gridDim.x)) {
    const GatingT *row = gating + static_cast<size_t>(token) * num_experts;
    float row_sum = 0.0f;

    for (int k_idx = 0; k_idx < topk; ++k_idx) {
      float best_val = -FLT_MAX;
      int best_idx = 0;
      for (int e = static_cast<int>(threadIdx.x); e < num_experts;
           e += static_cast<int>(blockDim.x)) {
        bool skip = false;
        for (int p = 0; p < k_idx; ++p) {
          if (selected[p] == e) {
            skip = true;
            break;
          }
        }
        if (skip) {
          continue;
        }
        float val = sigmoidf_fast(toFloat(row[e]));
        if (correction_bias != nullptr) {
          val += loadBias(correction_bias, bias_dtype_tag, e);
        }
        if (val > best_val || (val == best_val && e < best_idx)) {
          best_val = val;
          best_idx = e;
        }
      }

      s_val[threadIdx.x] = best_val;
      s_idx[threadIdx.x] = best_idx;
      __syncthreads();

      for (int offset = static_cast<int>(blockDim.x) / 2; offset > 0;
           offset >>= 1) {
        if (static_cast<int>(threadIdx.x) < offset) {
          float ov = s_val[threadIdx.x + offset];
          int oi = s_idx[threadIdx.x + offset];
          if (ov > s_val[threadIdx.x] ||
              (ov == s_val[threadIdx.x] && oi < s_idx[threadIdx.x])) {
            s_val[threadIdx.x] = ov;
            s_idx[threadIdx.x] = oi;
          }
        }
        __syncthreads();
      }

      if (threadIdx.x == 0) {
        int expert = s_idx[0];
        float val = s_val[0];
        if (correction_bias != nullptr) {
          val -= loadBias(correction_bias, bias_dtype_tag, expert);
        }
        s_weights[k_idx] = val;
        selected[k_idx] = expert;
        row_sum += val;
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      if (renormalize) {
        float inv = 1.0f / fmaxf(row_sum, 1e-20f);
        for (int k_idx = 0; k_idx < topk; ++k_idx) {
          s_weights[k_idx] *= inv;
        }
      }
      for (int k_idx = 0; k_idx < topk; ++k_idx) {
        const int out_i = token * topk + k_idx;
        topk_weights[out_i] = fromFloat<WeightT>(s_weights[k_idx]);
        topk_indices[out_i] = static_cast<IndexT>(selected[k_idx]);
      }
    }
    __syncthreads();
  }
}

template <typename GatingT, typename WeightT, typename IndexT>
void launchMoeTopkSigmoid(void *topk_weights, void *topk_indices,
                          const void *gating, const void *correction_bias,
                          int bias_dtype_tag, size_t tokens, size_t experts,
                          size_t topk, bool renormalize, void *stream) {
  if (tokens == 0) {
    return;
  }
  if (experts == 0 || topk == 0) {
    throw std::runtime_error(
        "QY moe_topk_sigmoid_launch: experts and topk must be > 0");
  }
  if (topk > static_cast<size_t>(kMaxTopk)) {
    throw std::runtime_error(
        "QY moe_topk_sigmoid_launch: topk exceeds kMaxTopk=" +
        std::to_string(kMaxTopk));
  }
  if (topk > experts) {
    throw std::runtime_error(
        "QY moe_topk_sigmoid_launch: topk must be <= experts");
  }

  constexpr int block = 128;
  // One block per token when possible; Denglin OORs on huge grids with large
  // blocks, but block=128 tolerates a higher grid than mul_scalar's old
  // 256/256.
  constexpr size_t kMaxGrid = 1024;
  int grid = static_cast<int>(std::min(tokens, kMaxGrid));
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  (void)cudaGetLastError();
  moeTopkSigmoidKernel<GatingT, WeightT, IndexT>
      <<<grid, block, 0, cuda_stream>>>(
          reinterpret_cast<const GatingT *>(gating),
          reinterpret_cast<WeightT *>(topk_weights),
          reinterpret_cast<IndexT *>(topk_indices), correction_bias,
          bias_dtype_tag, static_cast<int>(tokens), static_cast<int>(experts),
          static_cast<int>(topk), renormalize ? 1 : 0);
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("QY moe_topk_sigmoid_launch failed: ") +
        cudaGetErrorString(err) + " (tokens=" + std::to_string(tokens) +
        ", experts=" + std::to_string(experts) +
        ", topk=" + std::to_string(topk) + ", grid=" + std::to_string(grid) +
        ", block=" + std::to_string(block) + ")");
  }
}

template <typename GatingT, typename WeightT>
void launchWithIndex(void *topk_weights, void *topk_indices, const void *gating,
                     const void *correction_bias, int bias_dtype_tag,
                     size_t tokens, size_t experts, size_t topk,
                     bool renormalize, int index_dtype_tag, void *stream) {
  using infini::rt::DataType;
  switch (static_cast<DataType>(index_dtype_tag)) {
    case DataType::kInt32:
      launchMoeTopkSigmoid<GatingT, WeightT, int32_t>(
          topk_weights, topk_indices, gating, correction_bias, bias_dtype_tag,
          tokens, experts, topk, renormalize, stream);
      return;
    case DataType::kInt64:
      launchMoeTopkSigmoid<GatingT, WeightT, int64_t>(
          topk_weights, topk_indices, gating, correction_bias, bias_dtype_tag,
          tokens, experts, topk, renormalize, stream);
      return;
    default:
      throw std::runtime_error(
          "QY moe_topk_sigmoid_launch: unsupported index_dtype_tag " +
          std::to_string(index_dtype_tag));
  }
}

template <typename GatingT>
void launchWithWeight(void *topk_weights, void *topk_indices,
                      const void *gating, const void *correction_bias,
                      int bias_dtype_tag, size_t tokens, size_t experts,
                      size_t topk, bool renormalize, int weight_dtype_tag,
                      int index_dtype_tag, void *stream) {
  using infini::rt::DataType;
  switch (static_cast<DataType>(weight_dtype_tag)) {
    case DataType::kFloat16:
      launchWithIndex<GatingT, half>(
          topk_weights, topk_indices, gating, correction_bias, bias_dtype_tag,
          tokens, experts, topk, renormalize, index_dtype_tag, stream);
      return;
    case DataType::kBFloat16:
      launchWithIndex<GatingT, cuda_bfloat16>(
          topk_weights, topk_indices, gating, correction_bias, bias_dtype_tag,
          tokens, experts, topk, renormalize, index_dtype_tag, stream);
      return;
    case DataType::kFloat32:
      launchWithIndex<GatingT, float>(
          topk_weights, topk_indices, gating, correction_bias, bias_dtype_tag,
          tokens, experts, topk, renormalize, index_dtype_tag, stream);
      return;
    default:
      throw std::runtime_error(
          "QY moe_topk_sigmoid_launch: unsupported weight_dtype_tag " +
          std::to_string(weight_dtype_tag));
  }
}

}  // namespace

void moe_topk_sigmoid_launch(void *topk_weights, void *topk_indices,
                             const void *gating, const void *correction_bias,
                             size_t tokens, size_t experts, size_t topk,
                             bool renormalize, int gating_dtype_tag,
                             int weight_dtype_tag, int index_dtype_tag,
                             int bias_dtype_tag, void *stream) {
  using infini::rt::DataType;
  if (correction_bias != nullptr) {
    switch (static_cast<DataType>(bias_dtype_tag)) {
      case DataType::kFloat16:
      case DataType::kBFloat16:
      case DataType::kFloat32:
        break;
      default:
        throw std::runtime_error(
            "QY moe_topk_sigmoid_launch: unsupported bias_dtype_tag " +
            std::to_string(bias_dtype_tag));
    }
  }

  switch (static_cast<DataType>(gating_dtype_tag)) {
    case DataType::kFloat16:
      launchWithWeight<half>(topk_weights, topk_indices, gating,
                             correction_bias, bias_dtype_tag, tokens, experts,
                             topk, renormalize, weight_dtype_tag,
                             index_dtype_tag, stream);
      return;
    case DataType::kBFloat16:
      launchWithWeight<cuda_bfloat16>(
          topk_weights, topk_indices, gating, correction_bias, bias_dtype_tag,
          tokens, experts, topk, renormalize, weight_dtype_tag, index_dtype_tag,
          stream);
      return;
    case DataType::kFloat32:
      launchWithWeight<float>(topk_weights, topk_indices, gating,
                              correction_bias, bias_dtype_tag, tokens, experts,
                              topk, renormalize, weight_dtype_tag,
                              index_dtype_tag, stream);
      return;
    default:
      throw std::runtime_error(
          "QY moe_topk_sigmoid_launch: unsupported gating_dtype_tag " +
          std::to_string(gating_dtype_tag));
  }
}

}  // namespace infini::ops::qy
