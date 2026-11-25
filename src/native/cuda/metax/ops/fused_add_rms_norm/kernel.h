#ifndef INFINI_OPS_METAX_FUSED_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_METAX_FUSED_ADD_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/ops/fused_add_rms_norm/kernel.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/fused_add_rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<FusedAddRmsNorm, Device::Type::kMetax>
    : public CudaFusedAddRmsNorm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaFusedAddRmsNorm<Runtime<Device::Type::kMetax>>::CudaFusedAddRmsNorm;

  void operator()(Tensor input, Tensor residual,
                  const std::optional<Tensor> weight,
                  float epsilon) const override {
    if (num_tokens_ == 0) {
      return;
    }

    if (num_tokens_ > 128 || dim_ != 4096 ||
        input_strides_[input_strides_.size() - 2] % 8 != 0 ||
        residual_strides_[residual_strides_.size() - 2] % 8 != 0 ||
        reinterpret_cast<uintptr_t>(input.data()) % 16 != 0 ||
        reinterpret_cast<uintptr_t>(residual.data()) % 16 != 0 ||
        (weight.has_value() &&
         reinterpret_cast<uintptr_t>(weight->data()) % 16 != 0)) {
      CudaFusedAddRmsNorm<Runtime<Device::Type::kMetax>>::operator()(
          input, residual, weight, epsilon);
      return;
    }

    auto cuda_stream = static_cast<Runtime<Device::Type::kMetax>::Stream>(
        stream_ ? stream_ : 0);
    DispatchFunc<Device::Type::kMetax,
                 ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>>(
        input.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          auto weight_data = weight.has_value()
                                 ? reinterpret_cast<const T*>(weight->data())
                                 : nullptr;
          MetaxFusedAddRmsNormVectorizedKernel<T>
              <<<static_cast<uint32_t>(num_tokens_), 256, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(input.data()),
                  input_strides_[input_strides_.size() - 2],
                  reinterpret_cast<T*>(residual.data()),
                  residual_strides_[residual_strides_.size() - 2], weight_data,
                  epsilon_);
        },
        "MetaxFusedAddRmsNorm::operator()");
  }
};

}  // namespace infini::ops

#endif
