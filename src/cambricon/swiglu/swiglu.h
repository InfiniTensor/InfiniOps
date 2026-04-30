#ifndef INFINI_OPS_CAMBRICON_SWIGLU_SWIGLU_H_
#define INFINI_OPS_CAMBRICON_SWIGLU_SWIGLU_H_

#include "cambricon/common.h"
#include "base/swiglu.h"
#include "cambricon/data_type_.h"

namespace infini::ops {

template <typename T>
void SwigluUnion(void *workspace, int core_per_cluster, int cluster_count,
                 cnrtQueue_t queue, void *out, const void *input,
                 const void *gate, const size_t *out_shape,
                 const ptrdiff_t *out_strides, const size_t *input_shape,
                 const ptrdiff_t *input_strides, const size_t *gate_shape,
                 const ptrdiff_t *gate_strides, size_t output_size, int ndim,
                 bool fast_path, bool out_contiguous);

template <>
class Operator<Swiglu, Device::Type::kCambricon> : public Swiglu {
 public:
  Operator(const Tensor input, const Tensor gate, Tensor out)
      : Swiglu{input, gate, out} {
    cnrt_utils::GetLaunchConfig(input.device(), &core_per_cluster,
                                &cluster_count);
    cnrtMalloc(&default_workspace_, workspace_size_in_bytes());
  }

  void operator()(const Tensor input, const Tensor gate,
                  Tensor out) const override {
    auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);
    auto workspace{workspace_ ? workspace_ : default_workspace_};

    bool fast_path = is_input_contiguous_ && is_gate_contiguous_ &&
                     is_out_contiguous_ && input_shape_ == out_shape_ &&
                     gate_shape_ == out_shape_;

    DispatchFunc<List<DataType::kFloat16, DataType::kBFloat16,
                      DataType::kFloat32>>(
        {static_cast<int64_t>(out_type_)},
        [&](auto tag) {
          using T = TypeMapType<Device::Type::kCambricon, ListGet<0>(tag)>;
          SwigluUnion<T>(workspace, core_per_cluster, cluster_count, queue,
                         out.data(), input.data(), gate.data(),
                         out_shape_.data(), out_strides_.data(),
                         input_shape_.data(), input_strides_.data(),
                         gate_shape_.data(), gate_strides_.data(),
                         output_size_, ndim_, fast_path,
                         is_out_contiguous_);
        },
        "CambriconSwiglu::operator() - output dispatch");
  }

  ~Operator() { cnrtFree(default_workspace_); }

  std::size_t workspace_size_in_bytes() const override {
    return ndim_ * (3 * sizeof(size_t) + 3 * sizeof(ptrdiff_t));
  }

  void *default_workspace_{nullptr};
  int core_per_cluster = 0;
  int cluster_count = 0;
};

}  // namespace infini::ops

#endif
