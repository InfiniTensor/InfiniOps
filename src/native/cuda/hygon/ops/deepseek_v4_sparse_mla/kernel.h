#ifndef INFINI_OPS_HYGON_DEEPSEEK_V4_SPARSE_MLA_H_
#define INFINI_OPS_HYGON_DEEPSEEK_V4_SPARSE_MLA_H_

#include <cuda_runtime.h>
#include "native/cuda/hygon/flash_mla/bridge.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "base/deepseek_v4_sparse_mla.h"
#include "native/cuda/hygon/caster.cuh"

namespace infini::ops {
namespace v4_sparse_mla_detail {
constexpr size_t kDim = 512;
constexpr size_t kNope = 448;
constexpr size_t kValueBytes = 576;
constexpr unsigned kThreads = 128;

struct Pages {
  const uint8_t* data{};
  const int32_t* indices{};
  const int32_t* lengths{};
  size_t page_count{};
  size_t page_stride{};
  size_t page_tokens{};
  size_t index_width{};
};

// E4M3 finite-only encoding used by DeepSeek V4's packed cache.
static __device__ float decode_fp8(uint8_t byte) {
  const int exponent = (byte >> 3) & 15;
  const int mantissa = byte & 7;
  if (exponent == 15 && mantissa == 7) return NAN;
  const float value = exponent ? ldexpf(1.0f + mantissa * 0.125f, exponent - 7)
                               : ldexpf(static_cast<float>(mantissa), -9);
  return byte & 128 ? -value : value;
}

static __device__ float value_at(const uint8_t* values, const uint8_t* exponents, size_t channel) {
  if (channel < kNope) {
    return ldexpf(decode_fp8(values[channel]), static_cast<int>(exponents[channel / 64]) - 127);
  }
  return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(values + kNope)[channel - kNope]);
}

template <class T>
__global__ void AttendUnrounded(const T* q, const float* sink, Pages first, Pages second,
                       T* out, size_t rows, size_t heads, float scale) {
  __shared__ float reduce[kThreads];
  __shared__ float past_weight, new_weight, denominator;
  const unsigned lane = threadIdx.x;
  for (size_t block = blockIdx.x; block < rows * heads; block += gridDim.x) {
    const auto row = block / heads, head = block % heads;
    float accum[4] = {};
    float query[4];
    for (size_t i = 0; i < 4; ++i) {
      query[i] = Caster<Device::Type::kHygon>::template Cast<float>(q[block * kDim + i * kThreads + lane]);
    }
    const auto sink_score = sink[head];
    float maximum = sink_score;
    if (lane == 0) denominator = sink_score == -INFINITY ? 0.0f : 1.0f;
    for (int source = 0; source < 2; ++source) {
      const Pages table = source == 0 ? first : second;
      if (!table.data) continue;
      const int32_t length = table.lengths[row];
      const size_t positive = length <= 0 ? 0 : static_cast<size_t>(length);
      const size_t count = positive < table.index_width ? positive : table.index_width;
      for (size_t i = 0; i < count; ++i) {
        const int32_t id = table.indices[row * table.index_width + i];
        if (id < 0 || static_cast<size_t>(id) / table.page_tokens >= table.page_count) continue;
        const auto page = static_cast<size_t>(id) / table.page_tokens;
        const auto offset = static_cast<size_t>(id) % table.page_tokens;
        const auto* raw = table.data + page * table.page_stride;
        const auto* values = raw + offset * kValueBytes;
        const auto* exponents = raw + table.page_tokens * kValueBytes + offset * 8;
        float kv[4];
        float local = 0.0f;
        for (size_t j = 0; j < 4; ++j) {
          kv[j] = value_at(values, exponents, j * kThreads + lane);
          local += query[j] * kv[j];
        }
        reduce[lane] = local;
        __syncthreads();
        for (unsigned width = kThreads / 2; width; width /= 2) {
          if (lane < width) reduce[lane] += reduce[lane + width];
          __syncthreads();
        }
        if (lane == 0) {
          const float score = reduce[0] * scale;
          const float next = fmaxf(score, maximum);
          past_weight = maximum == -INFINITY ? 0.0f : expf(maximum - next);
          new_weight = next == INFINITY ? 0.0f : expf(score - next);
          if (maximum == INFINITY) past_weight = 1.0f;
          maximum = next;
          denominator = denominator * past_weight + new_weight;
        }
        __syncthreads();
        for (size_t j = 0; j < 4; ++j) {
          accum[j] = accum[j] * past_weight + kv[j] * new_weight;
        }
        __syncthreads();
      }
    }
    __syncthreads();
    for (size_t j = 0; j < 4; ++j) {
      out[block * kDim + j * kThreads + lane] = Caster<Device::Type::kHygon>::template Cast<T>(
          denominator ? accum[j] / denominator : 0.0f);
    }
    __syncthreads();
  }
}
constexpr size_t kMaxKeyTile = 128;

static __device__ bool valid_key(const Pages& table, int32_t id) {
  return id >= 0 && static_cast<size_t>(id) / table.page_tokens < table.page_count;
}

static __device__ void load_key(const Pages& table, int32_t id, unsigned lane, float (&kv)[4]) {
  const size_t page = static_cast<size_t>(id) / table.page_tokens;
  const size_t offset = static_cast<size_t>(id) % table.page_tokens;
  const auto* raw = table.data + page * table.page_stride;
  const auto* values = raw + offset * kValueBytes;
  const auto* exponents = raw + table.page_tokens * kValueBytes + offset * 8;
  for (size_t j = 0; j < 4; ++j) {
    kv[j] = value_at(values, exponents, j * kThreads + lane);
  }
}

template <class T>
__global__ void Attend(const T* q, const float* sink, Pages first, Pages second,
                       T* out, size_t rows, size_t heads, float scale, int parallel_queries) {
  using Convert = Caster<Device::Type::kHygon>;
  __shared__ float reduce[kThreads];
  __shared__ float scores[kMaxKeyTile];
  __shared__ float weights[kMaxKeyTile];
  __shared__ float prior_weight, tile_weight, denominator;
  const unsigned lane = threadIdx.x;
  const bool online_tiles = rows >= static_cast<size_t>(parallel_queries) && second.data;
  const size_t keys_per_tile = rows > 1 && !second.data ? 128 : 64;
  for (size_t block = blockIdx.x; block < rows * heads; block += gridDim.x) {
    const auto row = block / heads, head = block % heads;
    float accum[4] = {};
    float query[4];
    for (size_t j = 0; j < 4; ++j) {
      query[j] = Convert::template Cast<float>(q[block * kDim + j * kThreads + lane]);
    }
    float maximum = -INFINITY;
    if (lane == 0) denominator = 0.0f;
    for (int source = 0; source < 2; ++source) {
      const Pages table = source == 0 ? first : second;
      if (!table.data) continue;
      const int32_t length = table.lengths[row];
      const size_t positive = length <= 0 ? 0 : static_cast<size_t>(length);
      const size_t count = positive < table.index_width ? positive : table.index_width;
      const auto* indices = table.indices + row * table.index_width;
      for (size_t start = 0; start < count; start += keys_per_tile) {
        const size_t remaining = count - start;
        const size_t tile_size = remaining < keys_per_tile ? remaining : keys_per_tile;
        for (size_t i = 0; i < tile_size; ++i) {
          const int32_t id = indices[start + i];
          if (!valid_key(table, id)) {
            if (lane == 0) scores[i] = -INFINITY;
            continue;
          }
          float kv[4], dot = 0.0f;
          load_key(table, id, lane, kv);
          for (size_t j = 0; j < 4; ++j) dot += query[j] * kv[j];
          reduce[lane] = dot;
          __syncthreads();
          for (unsigned width = kThreads / 2; width; width /= 2) {
            if (lane < width) reduce[lane] += reduce[lane + width];
            __syncthreads();
          }
          if (lane == 0) scores[i] = reduce[0] * scale;
        }
        if (lane == 0) {
          float tile_max = -INFINITY;
          for (size_t i = 0; i < tile_size; ++i) tile_max = fmaxf(tile_max, scores[i]);
          const float next = fmaxf(maximum, tile_max);
          const float shift = online_tiles ? next : tile_max;
          float mass = 0.0f;
          for (size_t i = 0; i < tile_size; ++i) {
            const float score = scores[i];
            const float prob = score == -INFINITY ? 0.0f :
                (score == shift ? 1.0f : expf(score - shift));
            mass += prob;
            weights[i] = Convert::template Cast<float>(Convert::template Cast<T>(prob));
          }
          prior_weight = maximum == next ? 1.0f : expf(maximum - next);
          tile_weight = shift == next ? 1.0f : expf(shift - next);
          maximum = next;
          denominator = denominator * prior_weight + mass * tile_weight;
        }
        __syncthreads();
        float weighted[4] = {};
        for (size_t i = 0; i < tile_size; ++i) {
          const float weight = weights[i];
          if (weight == 0.0f) continue;
          float kv[4];
          load_key(table, indices[start + i], lane, kv);
          for (size_t j = 0; j < 4; ++j) weighted[j] += weight * kv[j];
        }
        for (size_t j = 0; j < 4; ++j) {
          accum[j] = accum[j] * prior_weight + weighted[j] * tile_weight;
        }
        __syncthreads();
      }
    }
    if (lane == 0 && denominator != 0.0f) {
      denominator += expf(sink[head] - maximum);
    }
    __syncthreads();
    for (size_t j = 0; j < 4; ++j) {
      out[block * kDim + j * kThreads + lane] = Convert::template Cast<T>(
          denominator ? accum[j] / denominator : 0.0f);
    }
    __syncthreads();
  }
}

}  // namespace v4_sparse_mla_detail

template <>
class Operator<DeepseekV4SparseMla, Device::Type::kHygon>
    : public DeepseekV4SparseMla {
 public:
  static bool supports(const Tensor q, const Tensor swa, const Tensor indices,
                       const Tensor lengths, const Tensor sink,
                       const std::optional<Tensor>& extra,
                       const std::optional<Tensor>& extra_indices,
                       const std::optional<Tensor>& extra_lengths,
                       int64_t extra_page_size, double scale, const Tensor out) {
    using namespace v4_sparse_mla_detail;
    if (q.device().type() != Device::Type::kHygon || q.ndim() != 3 || q.size(2) != kDim ||
        !q.size(0) || !q.size(1) || q.size(0) > std::numeric_limits<size_t>::max() / q.size(1) / kDim ||
        (q.dtype() != DataType::kBFloat16 && q.dtype() != DataType::kFloat32) ||
        out.dtype() != q.dtype() || out.shape() != q.shape() ||
        sink.dtype() != DataType::kFloat32 || sink.shape() != Tensor::Shape{q.size(1)} ||
        !std::isfinite(scale) || scale <= 0.0 || scale > std::numeric_limits<float>::max() ||
        static_cast<bool>(extra) != static_cast<bool>(extra_indices) ||
        static_cast<bool>(extra) != static_cast<bool>(extra_lengths))
      return false;
    const auto valid = [&](const Tensor& t) {
      return t.device() == q.device() && t.IsContiguous();
    };
    const auto pages_valid = [&](const Tensor& p, const Tensor& ids, const Tensor& lens, size_t page_tokens) {
      return p.dtype() == DataType::kUInt8 && p.ndim() == 2 && p.size(0) > 0 &&
             p.size(1) >= page_tokens * 584 && p.size(1) % 2 == 0 &&
             ids.dtype() == DataType::kInt32 && ids.ndim() == 2 && ids.size(0) == q.size(0) &&
             ids.size(1) > 0 && lens.dtype() == DataType::kInt32 && lens.shape() == Tensor::Shape{q.size(0)} &&
             valid(p) && valid(ids) && valid(lens);
    };
    return valid(q) && valid(sink) && valid(out) && pages_valid(swa, indices, lengths, 256) &&
           (!extra || ((extra_page_size == 64 || extra_page_size == 2) &&
                       pages_valid(*extra, *extra_indices, *extra_lengths, extra_page_size)));
  }

  Operator(const Tensor q, const Tensor swa, const Tensor indices,
           const Tensor lengths, const Tensor sink, const std::optional<Tensor> extra,
           const std::optional<Tensor> extra_indices, const std::optional<Tensor> extra_lengths,
           int64_t extra_page_size, double scale, Tensor out)
      : DeepseekV4SparseMla(q, swa, indices, lengths, sink, extra, extra_indices, extra_lengths, extra_page_size, scale, out) {
    if (!supports(q, swa, indices, lengths, sink, extra, extra_indices, extra_lengths, extra_page_size, scale, out))
      throw std::invalid_argument("DeepSeek V4 sparse MLA inputs are unsupported");
    if (cudaDeviceGetAttribute(&parallel_queries_, cudaDevAttrMultiProcessorCount, q.device().index()) != cudaSuccess)
      throw std::runtime_error("DeepSeek V4 sparse MLA device query failed");
#ifdef INFINIOPS_HAVE_HYGON_FLASHMLA
    vendor_ = q.dtype()==DataType::kBFloat16 && q.size(1)<=64 && q.size(0)<=3072;
    if (vendor_) workspace_bytes_=hygon_flash_mla::workspace_bytes(q.size(0),parallel_queries_);
#endif
  }

  bool capture_safe() const override { return true; }
  size_t workspace_size_in_bytes() const override { return workspace_bytes_; }

  void operator()(const Tensor q, const Tensor swa, const Tensor indices,
                  const Tensor lengths, const Tensor sink, const std::optional<Tensor> extra,
                  const std::optional<Tensor> extra_indices, const std::optional<Tensor> extra_lengths,
                  int64_t extra_page_size, double scale, Tensor out) const override {
    using namespace v4_sparse_mla_detail;
    if (!supports(q, swa, indices, lengths, sink, extra, extra_indices, extra_lengths, extra_page_size, scale, out))
      throw std::invalid_argument("DeepSeek V4 sparse MLA inputs changed");
    const auto page = [](const Tensor& p, const Tensor& ids, const Tensor& lens, size_t tokens) {
      return Pages{static_cast<const uint8_t*>(p.data()), static_cast<const int32_t*>(ids.data()),
                   static_cast<const int32_t*>(lens.data()), p.size(0), p.size(1), tokens, ids.size(1)};
    };
    const auto first = page(swa, indices, lengths, 256);
    const auto second = extra ? page(*extra, *extra_indices, *extra_lengths, extra_page_size) : Pages{};
#ifdef INFINIOPS_HAVE_HYGON_FLASHMLA
    if (vendor_) {
      if(!workspace_ || workspace_size_in_bytes_<workspace_bytes_)
        throw std::invalid_argument("DeepSeek V4 FlashMLA workspace changed");
      const auto convert=[](const Pages& p) {
        return hygon_flash_mla::Pages{p.data,p.indices,p.lengths,
          static_cast<int>(p.page_count),static_cast<int>(p.page_tokens),
          static_cast<int>(p.page_stride),static_cast<int>(p.index_width)};
      };
      hygon_flash_mla::run({q.data(),static_cast<const float*>(sink.data()),out.data(),
          workspace_,workspace_size_in_bytes_,stream_,static_cast<int>(q.size(0)),
          static_cast<int>(q.size(1)),parallel_queries_,static_cast<float>(scale),
          convert(first),convert(second)});
      return;
    }
#endif
    const size_t blocks = q.size(0) * q.size(1);
    const auto launch = [&](auto tag) {
      using T = decltype(tag);
      if constexpr (std::is_same_v<T, float>) {
      AttendUnrounded<T><<<static_cast<unsigned>(blocks < 65535 ? blocks : 65535), kThreads, 0,
                  static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const T*>(q.data()), static_cast<const float*>(sink.data()),
          first, second, static_cast<T*>(out.data()), q.size(0), q.size(1), static_cast<float>(scale));
      } else {
      Attend<T><<<static_cast<unsigned>(blocks < 65535 ? blocks : 65535), kThreads, 0,
                  static_cast<cudaStream_t>(stream_)>>>(
          static_cast<const T*>(q.data()), static_cast<const float*>(sink.data()),
          first, second, static_cast<T*>(out.data()), q.size(0), q.size(1), static_cast<float>(scale), parallel_queries_);
      }
    };
    if (q.dtype() == DataType::kFloat32) launch(float{});
    else launch(__nv_bfloat16{});
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("DeepSeek V4 sparse MLA launch failed");
  }
 private:
  int parallel_queries_{};
  bool vendor_{};
  size_t workspace_bytes_{};
};

}  // namespace infini::ops

#endif
