#ifndef INFINI_OPS_METAX_EMBEDDING_KERNEL_H_
#define INFINI_OPS_METAX_EMBEDDING_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/ops/embedding/kernel.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/embedding/kernel.h"

namespace infini::ops {

template <>
class Operator<Embedding, Device::Type::kMetax>
    : public CudaEmbedding<Runtime<Device::Type::kMetax>> {
 public:
  using CudaEmbedding<Runtime<Device::Type::kMetax>>::CudaEmbedding;

  using CudaEmbedding<Runtime<Device::Type::kMetax>>::operator();

  void operator()(const Tensor input, const Tensor weight,
                  const std::optional<int64_t> padding_idx,
                  const std::optional<double> max_norm, const double norm_type,
                  const bool scale_grad_by_freq, const bool sparse,
                  Tensor out) const override {
    if (num_indices_ == 0) {
      return;
    }

    if (max_norm.has_value() || !input.IsContiguous() || !out.IsContiguous() ||
        embedding_dim_ != 4096 || weight.stride(1) != 1 ||
        weight.stride(0) % 8 != 0 ||
        reinterpret_cast<uintptr_t>(weight.data()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(out.data()) % 16 != 0) {
      CudaEmbedding<Runtime<Device::Type::kMetax>>::operator()(
          input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq,
          sparse, out);
      return;
    }

    auto cuda_stream = static_cast<Runtime<Device::Type::kMetax>::Stream>(
        stream_ ? stream_ : 0);
    DispatchFunc<List<DataType::kInt32, DataType::kInt64>,
                 ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>>(
        {static_cast<int64_t>(input_dtype_),
         static_cast<int64_t>(weight_dtype_)},
        [&](auto list_tag) {
          using IndexT =
              TypeMapType<Device::Type::kMetax, ListGet<0>(list_tag)>;
          using T = TypeMapType<Device::Type::kMetax, ListGet<1>(list_tag)>;
          MetaxEmbeddingVectorizedKernel<T, IndexT>
              <<<num_indices_, 256, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()),
                  reinterpret_cast<const IndexT*>(input.data()),
                  reinterpret_cast<const T*>(weight.data()), weight.stride(0),
                  vocab_size_);
        },
        "MetaxEmbedding::operator()");
  }
};

}  // namespace infini::ops

#endif
