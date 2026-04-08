#ifndef INFINI_OPS_CAMBRICON_CAUSAL_SOFTMAX_H
#define INFINI_OPS_CAMBRICON_CAUSAL_SOFTMAX_H

#include "base/causal_softmax.h"
#include "cambricon/common.h"
#include "cambricon/data_type_.h"

namespace infini::ops {

// TODO: Remove forward declaration.
template <typename T>
void CausalSoftmaxUnion(void *workspace, int core_per_cluster,
                        int cluster_count, cnrtQueue_t queue, void *y,
                        const void *x, size_t batch_size_, size_t seq_len_,
                        size_t total_seq_len_, ptrdiff_t y_stride_b,
                        ptrdiff_t y_stride_i, ptrdiff_t y_stride_j,
                        ptrdiff_t x_stride_b, ptrdiff_t x_stride_i,
                        ptrdiff_t x_stride_j);

template <>
class Operator<CausalSoftmax, Device::Type::kCambricon> : public CausalSoftmax {
 public:
  Operator(const Tensor input, Tensor out) : CausalSoftmax{input, out} {
    cnrt_utils::GetLaunchConfig(input.device(), &core_per_cluster,
                                &cluster_count);
  }
  void operator()(const Tensor input, Tensor out) const override {
    auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);
    auto workspace{workspace_ ? workspace_ : default_workspace_};
    ptrdiff_t y_stride_b = ndim_ == 3 ? out_strides_[0] : 1;
    ptrdiff_t y_stride_i = ndim_ == 3 ? out_strides_[1] : out_strides_[0];
    ptrdiff_t y_stride_j = ndim_ == 3 ? out_strides_[2] : out_strides_[1];
    ptrdiff_t x_stride_b = ndim_ == 3 ? input_strides_[0] : 1;
    ptrdiff_t x_stride_i = ndim_ == 3 ? input_strides_[1] : input_strides_[0];
    ptrdiff_t x_stride_j = ndim_ == 3 ? input_strides_[2] : input_strides_[1];

    DispatchFunc<
        List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32>>(
        {static_cast<int64_t>(input.dtype())},
        [&](auto input_tag) {
          using InputT = infini::ops::TypeMapType<Device::Type::kCambricon, ListGet<0>(input_tag)>;
          CausalSoftmaxUnion<InputT>(
              workspace, core_per_cluster, cluster_count, queue, out.data(),
              input.data(), batch_size_, seq_len_, total_seq_len_, y_stride_b,
              y_stride_i, y_stride_j, x_stride_b, x_stride_i, x_stride_j);
        },
        "CambriconCausalSoftmax::operator() - output dispatch");
  }

  std::size_t workspace_size_in_bytes() const override { return 0; }

  ~Operator() {}

  void *default_workspace_{nullptr};
  int core_per_cluster = 0;
  int cluster_count = 0;
};

}  // namespace infini::ops

#endif
