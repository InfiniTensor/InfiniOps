#ifndef INFINI_OPS_CUDA_SILU_AND_MUL_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_SILU_AND_MUL_INFINILM_KERNEL_H_

#include <algorithm>
#include <cstddef>

#include "base/silu_and_mul_infinilm.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/silu_and_mul_infinilm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaSiluAndMulInfinilm : public SiluAndMulInfinilm {
 public:
  using SiluAndMulInfinilm::SiluAndMulInfinilm;

  void operator()(const Tensor input, Tensor out) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    int block_size = std::min(
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(), 1024);
    dim3 block(std::min(static_cast<Tensor::Size>(block_size), output_size_));
    dim3 grid(utils::CeilDiv(output_size_, block.x));

    DispatchFunc<AllFloatTypes, List<128, 256, 512, 1024>>(
        {static_cast<int64_t>(out_type_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          SiluAndMulInfinilmKernel<Backend::kDeviceType, T, kBlockSize>
              <<<grid, block, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()),
                  reinterpret_cast<const T*>(input.data()), output_size_,
                  hidden_size_);
        },
        "CudaSiluAndMulInfinilm::operator()");
  }
};

}  // namespace infini::ops

#endif
