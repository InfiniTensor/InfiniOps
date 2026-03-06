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
  using CausalSoftmax::CausalSoftmax;

  void operator()(void* stream, const Tensor input,
                  Tensor out) const override {
    auto cuda_stream =
        static_cast<typename Backend::stream_t>(stream ? stream : 0);

    auto stride_input_batch = ndim_ == 3 ? input_strides_[0] : 0;
    auto stride_input_row = input_strides_[ndim_ - 2];
    auto stride_out_batch = ndim_ == 3 ? out_strides_[0] : 0;
    auto stride_out_row = out_strides_[ndim_ - 2];

    dim3 grid(static_cast<unsigned>(seq_len_),
              static_cast<unsigned>(batch_size_));

    if (out.dtype() != input.dtype()) {
      std::abort();
    }

    DispatchFunc<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>(
        out.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          CausalSoftmaxKernel<causal_softmax::kBlockSize, T, float>
              <<<grid, causal_softmax::kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()),
                  reinterpret_cast<const T*>(input.data()), batch_size_,
                  seq_len_, total_seq_len_, stride_out_batch, stride_out_row,
                  stride_input_batch, stride_input_row);
        },
        "CudaCausalSoftmax::operator()");
  }
};

}  // namespace infini::ops

#endif
