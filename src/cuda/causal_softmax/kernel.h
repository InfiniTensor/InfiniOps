#ifndef INFINI_OPS_CUDA_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_CUDA_CAUSAL_SOFTMAX_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "base/causal_softmax.h"
#include "cuda/causal_softmax/kernel.cuh"
#include "cuda/kernel_commons.h"
#include "data_type.h"
#include "dispatcher.h"

namespace infini::ops {

namespace causal_softmax::detail {

template <typename Backend, typename = void>
struct MaxBlockSize : std::integral_constant<int, CUDA_BLOCK_SIZE_2048> {};

template <typename Backend>
struct MaxBlockSize<Backend, std::void_t<decltype(Backend::max_block_size)>>
    : std::integral_constant<int, Backend::max_block_size> {};

}  // namespace causal_softmax::detail

template <typename Backend>
class CudaCausalSoftmax : public CausalSoftmax {
 public:
  using CausalSoftmax::CausalSoftmax;

  void operator()(const Tensor input, Tensor out) const override {
    auto cuda_stream =
        static_cast<typename Backend::stream_t>(stream_ ? stream_ : 0);

    auto stride_input_batch = ndim_ == 3 ? input_strides_[0] : 0;
    auto stride_input_row = input_strides_[ndim_ - 2];
    auto stride_out_batch = ndim_ == 3 ? out_strides_[0] : 0;
    auto stride_out_row = out_strides_[ndim_ - 2];

    dim3 grid(static_cast<unsigned>(seq_len_),
              static_cast<unsigned>(batch_size_));

    if (out.dtype() != input.dtype()) {
      std::abort();
    }

    constexpr int kMaxBlockSize =
        causal_softmax::detail::MaxBlockSize<Backend>::value;
    int block_size = std::min(GetOptimalBlockSize(), kMaxBlockSize);

    DispatchFunc<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>(
        out.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;

#define LAUNCH_CAUSAL_SOFTMAX_KERNEL(BLOCK_SIZE)                           \
  CausalSoftmaxKernel<BLOCK_SIZE, T, float>                                \
      <<<grid, BLOCK_SIZE, 0, cuda_stream>>>(                              \
          reinterpret_cast<T*>(out.data()),                                \
          reinterpret_cast<const T*>(input.data()), batch_size_, seq_len_, \
          total_seq_len_, stride_out_batch, stride_out_row,                \
          stride_input_batch, stride_input_row);

          if constexpr (kMaxBlockSize >= CUDA_BLOCK_SIZE_2048) {
            if (block_size == CUDA_BLOCK_SIZE_2048) {
              LAUNCH_CAUSAL_SOFTMAX_KERNEL(CUDA_BLOCK_SIZE_2048)
              return;
            }
          }
          if constexpr (kMaxBlockSize >= CUDA_BLOCK_SIZE_1024) {
            if (block_size == CUDA_BLOCK_SIZE_1024) {
              LAUNCH_CAUSAL_SOFTMAX_KERNEL(CUDA_BLOCK_SIZE_1024)
              return;
            }
          }
          if constexpr (kMaxBlockSize >= CUDA_BLOCK_SIZE_512) {
            if (block_size == CUDA_BLOCK_SIZE_512) {
              LAUNCH_CAUSAL_SOFTMAX_KERNEL(CUDA_BLOCK_SIZE_512)
              return;
            }
          }
          if constexpr (kMaxBlockSize >= CUDA_BLOCK_SIZE_256) {
            if (block_size == CUDA_BLOCK_SIZE_256) {
              LAUNCH_CAUSAL_SOFTMAX_KERNEL(CUDA_BLOCK_SIZE_256)
              return;
            }
          }
          LAUNCH_CAUSAL_SOFTMAX_KERNEL(CUDA_BLOCK_SIZE_128)

#undef LAUNCH_CAUSAL_SOFTMAX_KERNEL
        },
        "CudaCausalSoftmax::operator()");
  }
};

}  // namespace infini::ops

#endif
