#ifndef INFINI_OPS_CPU_RMS_NORM_DSL_H_
#define INFINI_OPS_CPU_RMS_NORM_DSL_H_

#include "cpu/templates/reduce_transform.h"
#include "base/rms_norm.h"
#include "impl.h"
#include "cpu/rms_norm/registry.h"

namespace infini::ops {

// CPU reduce op for `RmsNorm` (DSL).
struct DslCpuRmsNormReduce {
  float Init() const { return 0.f; }

  float Accumulate(float acc, float v) const { return acc + v * v; }

  float Finalize(float acc, size_t count) const {
    return 1.f / std::sqrt(acc / static_cast<float>(count) + epsilon);
  }

  float epsilon;
};

// CPU transform op for `RmsNorm` (DSL).
struct DslCpuRmsNormTransform {
  template <typename T>
  T Apply(T x, float reduced, size_t i) const {
    const auto* w = static_cast<const T*>(weight);

    return Caster<Device::Type::kCpu>::Cast<T>(
        Caster<Device::Type::kCpu>::Cast<float>(x) *
        Caster<Device::Type::kCpu>::Cast<float>(w[i]) * reduced);
  }

  const void* weight;
};

template <>
class Operator<RmsNorm, Device::Type::kCpu, Impl::kDsl> : public RmsNorm {
 public:
  using RmsNorm::RmsNorm;

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    CpuReduceThenTransform<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>>(
        input, out, batch_size_, nhead_, dim_,
        out.dtype(), input_strides_, out_strides_,
        DslCpuRmsNormReduce{eps},
        DslCpuRmsNormTransform{weight.data()});
  }
};

}  // namespace infini::ops

#endif
