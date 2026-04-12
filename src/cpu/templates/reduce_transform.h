#ifndef INFINI_OPS_CPU_TEMPLATES_REDUCE_TRANSFORM_H_
#define INFINI_OPS_CPU_TEMPLATES_REDUCE_TRANSFORM_H_

#include <cmath>
#include <cstddef>

#include "cpu/caster_.h"
#include "dispatcher.h"
#include "tensor.h"

namespace infini::ops {

// CPU reduce-then-transform brick.
//
// Iterates over [batch, head] slices. For each slice, reduces over `dim`
// elements, then applies a transform using the reduction result.
//
// `ReduceOp` must define:
//   `float Init()` — identity element.
//   `float Accumulate(float acc, float value)` — fold one element.
//   `float Finalize(float acc, size_t count)` — post-process total.
//
// `TransformOp` must define:
//   `T Apply(T x, float reduced, size_t i)` — per-element transform.
template <typename TypeList, typename ReduceOp, typename TransformOp>
void CpuReduceThenTransform(
    const Tensor in, Tensor out, size_t batch_size, size_t nhead,
    size_t dim, DataType dtype, const Tensor::Strides& in_strides,
    const Tensor::Strides& out_strides, ReduceOp reduce_op,
    TransformOp transform_op) {
  auto stride_in_batch = in_strides.size() > 1 ? in_strides[0] : 0;
  auto stride_in_head =
      in_strides.size() > 1 ? in_strides[1] : in_strides[0];
  auto stride_out_batch = out_strides.size() > 1 ? out_strides[0] : 0;
  auto stride_out_head =
      out_strides.size() > 1 ? out_strides[1] : out_strides[0];

  DispatchFunc<Device::Type::kCpu, TypeList>(
      dtype,
      [&](auto tag) {
        using T = typename decltype(tag)::type;

        const auto* in_ptr = static_cast<const T*>(in.data());
        auto* out_ptr = static_cast<T*>(out.data());

        for (size_t bi = 0; bi < batch_size; ++bi) {

          for (size_t hi = 0; hi < nhead; ++hi) {
            auto in_row = in_ptr + bi * stride_in_batch + hi * stride_in_head;
            auto out_row =
                out_ptr + bi * stride_out_batch + hi * stride_out_head;

            // Reduction phase.
            float acc = reduce_op.Init();

            for (size_t k = 0; k < dim; ++k) {
              float v = Caster<Device::Type::kCpu>::Cast<float>(in_row[k]);
              acc = reduce_op.Accumulate(acc, v);
            }

            float reduced = reduce_op.Finalize(acc, dim);

            // Transform phase.
            for (size_t k = 0; k < dim; ++k) {
              out_row[k] =
                  transform_op.template Apply<T>(in_row[k], reduced, k);
            }
          }
        }
      },
      "CpuReduceThenTransform");
}

// ---------- Built-in ops matching the CUDA counterparts ---------------------

struct CpuMeanSquareReduce {
  float Init() const { return 0.f; }

  float Accumulate(float acc, float v) const { return acc + v * v; }

  float Finalize(float acc, size_t count) const {
    return 1.f / std::sqrt(acc / static_cast<float>(count) + epsilon);
  }

  float epsilon;
};

struct CpuRmsNormTransform {
  template <typename T>
  T Apply(T x, float rms, size_t i) const {
    const auto* w = static_cast<const T*>(weight);

    return Caster<Device::Type::kCpu>::Cast<T>(
        Caster<Device::Type::kCpu>::Cast<float>(x) *
        Caster<Device::Type::kCpu>::Cast<float>(w[i]) * rms);
  }

  const void* weight;
};

}  // namespace infini::ops

#endif
