#ifndef INFINI_OPS_CUDA_RMS_NORM_DSL_H_
#define INFINI_OPS_CUDA_RMS_NORM_DSL_H_

#include "cuda/templates/reduce_transform.cuh"
#include "base/rms_norm.h"

namespace infini::ops {

// Reduce op for `RmsNorm` (DSL).
struct DslRmsNormReduce {
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

// Transform op for `RmsNorm` (DSL).
struct DslRmsNormTransform {
  template <Device::Type kDev, typename TData>
  __device__ __forceinline__ TData Apply(TData x, float reduced,
                                         size_t i) const {
    return Caster<kDev>::template Cast<TData>(
        Caster<kDev>::template Cast<float>(x) *
        Caster<kDev>::template Cast<float>(static_cast<const TData*>(weight)[i]) * reduced);
  }

  const void* weight;
};

template <typename Backend>
class DslCudaRmsNorm : public RmsNorm {
 public:
  using RmsNorm::RmsNorm;

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    LaunchReduceThenTransform<Backend, ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>>(
        stream_, input, out, batch_size_, nhead_, dim_,
        out.dtype(), input_strides_, out_strides_,
        DslRmsNormReduce{eps},
        DslRmsNormTransform{weight.data()});
  }
};

}  // namespace infini::ops

#endif
