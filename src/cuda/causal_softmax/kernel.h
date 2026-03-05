#ifndef INFINI_OPS_CUDA_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_CUDA_CAUSAL_SOFTMAX_KERNEL_H_

#include <cstdint>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "base/causal_softmax.h"
#include "cuda/causal_softmax/kernel.cuh"
#include "data_type.h"
#include "dispatcher.h"

namespace infini::ops {

namespace causal_softmax {

constexpr unsigned int kBlockSize = 256;

}  // namespace causal_softmax

template <typename Backend>
class CudaCausalSoftmax : public CausalSoftmax {
 public:
  CudaCausalSoftmax(const Tensor y, const Tensor x) : CausalSoftmax{y, x} {}

  void operator()(void* stream, Tensor y, const Tensor x) const override {
    auto cuda_stream =
        static_cast<typename Backend::stream_t>(stream ? stream : 0);

    auto y_stride_b = ndim_ == 3 ? y_strides_[0] : 0;
    auto y_stride_i = y_strides_[ndim_ - 2];
    auto x_stride_b = ndim_ == 3 ? x_strides_[0] : 0;
    auto x_stride_i = x_strides_[ndim_ - 2];

    if (y.dtype() != x.dtype()) {
      std::abort();
    }

    dim3 grid(static_cast<unsigned>(seq_len_),
              static_cast<unsigned>(batch_size_));

    DispatchFunc<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>(
        y.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          CausalSoftmaxKernel<causal_softmax::kBlockSize, T, float>
              <<<grid, causal_softmax::kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(y.data()),
                  reinterpret_cast<const T*>(x.data()), batch_size_, seq_len_,
                  total_seq_len_, y_stride_b, y_stride_i, x_stride_b,
                  x_stride_i);
        },
        "CudaCausalSoftmax::operator()");
  }
};

}  // namespace infini::ops

#endif
