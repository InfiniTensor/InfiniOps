#ifndef INFINI_OPS_CUDA_CLOVER_BLOCK_CONV1D_KERNEL_H_
#define INFINI_OPS_CUDA_CLOVER_BLOCK_CONV1D_KERNEL_H_

#include "base/clover_block_conv1d.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/clover_block_conv1d/kernel.cuh"

namespace infini::ops {

template <typename Backend>
class CudaCloverBlockConv1d : public CloverBlockConv1d {
 public:
  using CloverBlockConv1d::CloverBlockConv1d;

  void operator()(const Tensor input, const Tensor state, const Tensor indices,
                  const Tensor weight, const Tensor right_weight,
                  const Tensor valid, size_t block_size,
                  Tensor out) const override {
    const auto tokens = input.size(0);
    const auto channels = input.size(1);
    const auto requests = indices.numel();
    if (tokens == 0 || channels == 0 || requests == 0) return;

    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    constexpr unsigned int kThreads = 128;
    const unsigned int ch_blocks =
        static_cast<unsigned int>((channels + kThreads - 1) / kThreads);
    constexpr int kTileTokens = 4;
    const int num_tiles =
        static_cast<int>((block_size + kTileTokens - 1) / kTileTokens);
    const unsigned int programs =
        static_cast<unsigned int>(requests * num_tiles);

    DispatchFunc<ReducedFloatTypes, List<DataType::kInt32, DataType::kInt64>>(
        {static_cast<int64_t>(input.dtype()),
         static_cast<int64_t>(indices.dtype())},
        [&](auto list_tag) {
          using Data = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using Index = TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
          CloverBlockConv1dKernel<Backend::kDeviceType, Data, Index,
                                  kTileTokens>
              <<<dim3(ch_blocks, programs), kThreads, 0, cuda_stream>>>(
                  reinterpret_cast<const Data*>(input.data()),
                  reinterpret_cast<const Data*>(state.data()),
                  reinterpret_cast<const Index*>(indices.data()),
                  reinterpret_cast<const Data*>(weight.data()),
                  reinterpret_cast<const Data*>(right_weight.data()),
                  reinterpret_cast<const int8_t*>(valid.data()),
                  reinterpret_cast<Data*>(out.data()),
                  static_cast<int64_t>(channels), static_cast<int>(block_size),
                  static_cast<int>(state.size(0)),
                  static_cast<int64_t>(input.stride(0)),
                  static_cast<int64_t>(input.stride(1)),
                  static_cast<int64_t>(state.stride(0)),
                  static_cast<int64_t>(state.stride(1)),
                  static_cast<int64_t>(state.stride(2)),
                  static_cast<int64_t>(weight.stride(0)),
                  static_cast<int64_t>(weight.stride(2)),
                  static_cast<int64_t>(right_weight.stride(0)),
                  static_cast<int64_t>(right_weight.stride(2)),
                  static_cast<int64_t>(out.stride(0)),
                  static_cast<int64_t>(out.stride(1)), num_tiles);
        },
        "CudaCloverBlockConv1d::operator()");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_CLOVER_BLOCK_CONV1D_KERNEL_H_
