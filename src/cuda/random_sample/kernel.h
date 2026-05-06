#ifndef INFINI_OPS_CUDA_RANDOM_SAMPLE_KERNEL_H_
#define INFINI_OPS_CUDA_RANDOM_SAMPLE_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "base/random_sample.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"

// FlashInfer sampling kernels — included directly, same as cuBLAS/cuBLASLt.
#include "flashinfer/sampling.cuh"

namespace infini::ops {

template <typename InT>
__global__ void CastToFloatKernel(float* __restrict__ out,
                                  const InT* __restrict__ in, uint32_t n) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = static_cast<float>(in[idx]);
  }
}

template <typename Backend>
class CudaRandomSample : public RandomSample {
 public:
  CudaRandomSample(const Tensor logits, Tensor out, Tensor valid,
                   std::optional<Tensor> temperature, float temperature_val,
                   std::optional<Tensor> top_k, int top_k_val,
                   std::optional<Tensor> top_p, float top_p_val,
                   std::optional<Tensor> min_p, float min_p_val,
                   std::uint64_t seed, std::uint64_t offset,
                   bool deterministic)
      : RandomSample(logits, out, valid, temperature, temperature_val, top_k,
                     top_k_val, top_p, top_p_val, min_p, min_p_val, seed,
                     offset, deterministic) {
    AllocateBuffers();
  }

  CudaRandomSample(const Tensor logits, Tensor out, Tensor valid,
                   std::uint64_t seed, std::uint64_t offset)
      : RandomSample(logits, out, valid, seed, offset) {
    AllocateBuffers();
  }

  ~CudaRandomSample() {
    Backend::Free(d_float_logits_);
    Backend::Free(d_probs_);
    if (d_workspace_) Backend::Free(d_workspace_);
    Backend::Free(d_top_k_float_);
  }

  void operator()(const Tensor logits, Tensor out, Tensor valid,
                  std::optional<Tensor> temperature, float temperature_val,
                  std::optional<Tensor> top_k, int top_k_val,
                  std::optional<Tensor> top_p, float top_p_val,
                  std::optional<Tensor> min_p, float min_p_val,
                  std::uint64_t seed, std::uint64_t offset,
                  bool deterministic) const override {
    auto stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    DispatchFunc<ConcatType<FloatTypes, ReducedFloatTypes>,
                 List<DataType::kInt32, DataType::kInt64>>(
        {static_cast<int64_t>(logits_dtype_),
         static_cast<int64_t>(out_dtype_)},
        [&](auto list_tag) {
          using T =
              TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using IdType =
              TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
          Compute<T, IdType>(stream, logits, out, valid, temperature,
                             temperature_val, top_k, top_k_val, top_p,
                             top_p_val, min_p, min_p_val, seed, offset,
                             deterministic);
        },
        "CudaRandomSample::operator()");
  }

  void operator()(const Tensor logits, Tensor out, Tensor valid,
                  std::uint64_t seed,
                  std::uint64_t offset) const override {
    return operator()(logits, out, valid, temperature_, temperature_val_,
                      top_k_, top_k_val_, top_p_, top_p_val_, min_p_,
                      min_p_val_, seed, offset, deterministic_);
  }

 private:
  void* d_float_logits_{nullptr};
  void* d_probs_{nullptr};
  void* d_workspace_{nullptr};
  void* d_top_k_float_{nullptr};
  size_t workspace_size_{0};

  void AllocateBuffers() {
    auto total = static_cast<size_t>(batch_size_) * vocab_size_;
    Backend::Malloc(&d_float_logits_, total * sizeof(float));
    Backend::Malloc(&d_probs_, total * sizeof(float));

    constexpr uint32_t kSliceSize = 8192;
    uint32_t num_slices = (vocab_size_ + kSliceSize - 1) / kSliceSize;
    workspace_size_ =
        static_cast<size_t>(batch_size_) * num_slices * 8;  // PartialSoftmaxResult
    Backend::Malloc(&d_workspace_, std::max(workspace_size_, size_t(1)));

    Backend::Malloc(&d_top_k_float_,
                    static_cast<size_t>(batch_size_) * sizeof(float));
  }

  std::pair<float*, float> ResolveFloatParam(std::optional<Tensor> t,
                                             float scalar_val) const {
    if (t.has_value()) {
      assert(t->dtype() == DataType::kFloat32 &&
             "per-batch float param must be float32 for CUDA backend");
      return {const_cast<float*>(static_cast<const float*>(t->data())), scalar_val};
    }
    return {nullptr, scalar_val};
  }

  int GetIntParam(std::optional<Tensor> t, int scalar_val,
                  Tensor::Size batch_idx) const {
    if (t.has_value()) {
      const auto& tensor = *t;
      auto stride = tensor.strides().empty() ? 1 : tensor.strides()[0];
      auto off = batch_idx * stride;
      switch (tensor.dtype()) {
        case DataType::kInt32:
          return static_cast<const int32_t*>(tensor.data())[off];
        case DataType::kInt64:
          return static_cast<int>(
              static_cast<const int64_t*>(tensor.data())[off]);
        default:
          assert(false && "unsupported dtype for int param");
          return scalar_val;
      }
    }
    return scalar_val;
  }

  template <typename T, typename IdType>
  void Compute(typename Backend::Stream stream, const Tensor logits,
               Tensor out, Tensor valid,
               std::optional<Tensor> temperature, float temperature_val,
               std::optional<Tensor> top_k, int top_k_val,
               std::optional<Tensor> top_p, float top_p_val,
               std::optional<Tensor> min_p, float min_p_val,
               std::uint64_t seed, std::uint64_t offset,
               bool deterministic) const {
    auto* float_logits = static_cast<float*>(d_float_logits_);
    auto* probs = static_cast<float*>(d_probs_);
    auto* out_ptr = static_cast<IdType*>(out.data());
    auto* valid_ptr = reinterpret_cast<bool*>(valid.data());
    auto total = static_cast<uint32_t>(batch_size_ * vocab_size_);

    // --- Step 1: logits -> float32 probs ---
    float* temp_arr = nullptr;
    if (temperature.has_value()) {
      assert(temperature->dtype() == DataType::kFloat32 &&
             "temperature tensor must be float32 for CUDA backend");
      temp_arr = const_cast<float*>(static_cast<const float*>(temperature->data()));
    }

    if constexpr (!std::is_same_v<T, float>) {
      auto* logits_ptr = static_cast<const T*>(logits.data());
      constexpr int kBlock = 256;
      int grid = (total + kBlock - 1) / kBlock;
      CastToFloatKernel<T>
          <<<grid, kBlock, 0, stream>>>(float_logits, logits_ptr, total);
    }

    {
      float* softmax_in = (std::is_same_v<T, float>)
                               ? const_cast<float*>(static_cast<const float*>(logits.data()))
                               : float_logits;
      flashinfer::sampling::OnlineSoftmax<float>(
          softmax_in, probs, batch_size_, vocab_size_,
          temp_arr, temperature_val,
          d_workspace_, workspace_size_, false, stream);
    }

    // --- Step 2: resolve per-batch parameters ---
    auto [top_p_arr, top_p_scalar] = ResolveFloatParam(top_p, top_p_val);
    auto [min_p_arr, min_p_scalar] = ResolveFloatParam(min_p, min_p_val);

    float* top_k_float_arr = nullptr;
    IdType* top_k_id_arr = nullptr;
    if (top_k.has_value()) {
      const auto& tk = *top_k;
      auto stride = tk.strides().empty() ? 1 : tk.strides()[0];

      // Copy int values from device to host, then convert to float.
      std::vector<float> host_top_k(batch_size_);
      if (tk.dtype() == DataType::kInt32) {
        std::vector<int32_t> host_buf(batch_size_);
        Backend::Memcpy(host_buf.data(), tk.data(),
                        static_cast<size_t>(batch_size_) * sizeof(int32_t),
                        Backend::MemcpyDeviceToHost);
        for (Tensor::Size i = 0; i < batch_size_; ++i)
          host_top_k[i] = static_cast<float>(host_buf[i * stride]);
      } else {
        std::vector<int64_t> host_buf(batch_size_);
        Backend::Memcpy(host_buf.data(), tk.data(),
                        static_cast<size_t>(batch_size_) * sizeof(int64_t),
                        Backend::MemcpyDeviceToHost);
        for (Tensor::Size i = 0; i < batch_size_; ++i)
          host_top_k[i] = static_cast<float>(host_buf[i * stride]);
      }

      Backend::Memcpy(d_top_k_float_, host_top_k.data(),
                      static_cast<size_t>(batch_size_) * sizeof(float),
                      Backend::MemcpyHostToDevice);
      top_k_float_arr = static_cast<float*>(d_top_k_float_);

      if ((tk.dtype() == DataType::kInt32 && sizeof(IdType) == 4) ||
          (tk.dtype() == DataType::kInt64 && sizeof(IdType) == 8)) {
        top_k_id_arr = const_cast<IdType*>(static_cast<const IdType*>(tk.data()));
      }
    }

    // --- Step 3: dispatch to FlashInfer sampling function ---
    bool use_top_k = top_k_val > 0 || top_k.has_value();
    bool use_top_p =
        (top_p_val > 0.f && top_p_val < 1.f) || top_p.has_value();
    bool use_min_p = min_p_val > 0.f || min_p.has_value();

    namespace fis = flashinfer::sampling;

    if (use_top_k && use_top_p) {
      fis::TopKTopPSamplingFromProb<float, IdType>(
          probs, top_k_id_arr, top_p_arr,
          out_ptr, valid_ptr, /*indices=*/nullptr,
          batch_size_, static_cast<IdType>(top_k_val), top_p_scalar,
          vocab_size_, deterministic,
          /*seed_arr=*/nullptr, seed, /*offset_arr=*/nullptr, offset, stream);
    } else if (use_top_k) {
      fis::TopKSamplingFromProb<float, IdType>(
          probs, out_ptr, valid_ptr, /*indices=*/nullptr,
          top_k_float_arr, batch_size_,
          static_cast<uint32_t>(top_k_val), vocab_size_, deterministic,
          /*seed_arr=*/nullptr, seed, /*offset_arr=*/nullptr, offset, stream);
    } else if (use_top_p) {
      fis::TopPSamplingFromProb<float, IdType>(
          probs, out_ptr, valid_ptr, /*indices=*/nullptr,
          top_p_arr, batch_size_, top_p_scalar,
          vocab_size_, deterministic,
          /*seed_arr=*/nullptr, seed, /*offset_arr=*/nullptr, offset, stream);
    } else if (use_min_p) {
      fis::MinPSamplingFromProb<float, IdType>(
          probs, min_p_arr, out_ptr, valid_ptr, /*indices=*/nullptr,
          batch_size_, min_p_scalar,
          vocab_size_, deterministic,
          /*seed_arr=*/nullptr, seed, /*offset_arr=*/nullptr, offset, stream);
    } else {
      fis::SamplingFromProb<float, IdType>(
          probs, out_ptr, valid_ptr, /*indices=*/nullptr,
          batch_size_, vocab_size_, deterministic,
          /*seed_arr=*/nullptr, seed, /*offset_arr=*/nullptr, offset, stream);
    }
  }
};

}  // namespace infini::ops

#endif
